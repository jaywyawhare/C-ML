#include "backend/remote_device.h"
#include "backend/device.h"
#include "core/logging.h"

#include <stdio.h>
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#ifdef __linux__
#include <dlfcn.h>
#define CML_HAS_DLFCN 1
#endif

/* Wire format
 * Request:  [uint32 opcode][uint32 payload_size][payload...]
 * Response: [uint32 status][uint32 payload_size][payload...]
 */

typedef struct {
    uint32_t opcode;
    uint32_t payload_size;
} MsgHeader;

typedef struct {
    uint32_t status;
    uint32_t payload_size;
} RespHeader;

/* Reliable send/recv helpers */

static int send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = (const uint8_t*)buf;
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = send(fd, p + sent, len - sent, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return -1;
        }
        sent += (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void* buf, size_t len) {
    uint8_t* p = (uint8_t*)buf;
    size_t got = 0;
    while (got < len) {
        ssize_t n = recv(fd, p + got, len - got, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return -1;
        }
        got += (size_t)n;
    }
    return 0;
}

static int send_msg(int fd, uint32_t opcode, const void* payload, uint32_t payload_size) {
    MsgHeader hdr = {.opcode = htonl(opcode), .payload_size = htonl(payload_size)};
    if (send_all(fd, &hdr, sizeof(hdr)) != 0) return -1;
    if (payload_size > 0 && payload) {
        if (send_all(fd, payload, payload_size) != 0) return -1;
    }
    return 0;
}

static int recv_resp(int fd, uint32_t* status, void* payload, uint32_t* payload_size) {
    RespHeader rhdr;
    if (recv_all(fd, &rhdr, sizeof(rhdr)) != 0) return -1;
    *status = ntohl(rhdr.status);
    uint32_t psize = ntohl(rhdr.payload_size);
    if (payload_size) *payload_size = psize;
    if (psize > 0 && payload) {
        if (recv_all(fd, payload, psize) != 0) return -1;
    } else if (psize > 0) {
        uint8_t discard[4096];
        uint32_t remaining = psize;
        while (remaining > 0) {
            uint32_t chunk = remaining < sizeof(discard) ? remaining : sizeof(discard);
            if (recv_all(fd, discard, chunk) != 0) return -1;
            remaining -= chunk;
        }
    }
    return 0;
}

static int send_resp(int fd, uint32_t status, const void* payload, uint32_t payload_size) {
    RespHeader rhdr = {.status = htonl(status), .payload_size = htonl(payload_size)};
    if (send_all(fd, &rhdr, sizeof(rhdr)) != 0) return -1;
    if (payload_size > 0 && payload) {
        if (send_all(fd, payload, payload_size) != 0) return -1;
    }
    return 0;
}

/* Client */

CMLRemoteDevice* cml_remote_connect(const char* host, int port) {
    if (!host || port <= 0) return NULL;

    CMLRemoteDevice* dev = (CMLRemoteDevice*)calloc(1, sizeof(CMLRemoteDevice));
    if (!dev) return NULL;

    strncpy(dev->host, host, sizeof(dev->host) - 1);
    dev->port = port;

    struct addrinfo hints = {0}, *res = NULL;
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    if (getaddrinfo(host, port_str, &hints, &res) != 0 || !res) {
        LOG_ERROR("remote_connect: cannot resolve %s:%d", host, port);
        free(dev);
        return NULL;
    }

    dev->sock_fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (dev->sock_fd < 0) {
        LOG_ERROR("remote_connect: socket() failed: %s", strerror(errno));
        freeaddrinfo(res);
        free(dev);
        return NULL;
    }

    int flag = 1;
    setsockopt(dev->sock_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

    if (connect(dev->sock_fd, res->ai_addr, res->ai_addrlen) != 0) {
        LOG_ERROR("remote_connect: connect to %s:%d failed: %s", host, port, strerror(errno));
        close(dev->sock_fd);
        freeaddrinfo(res);
        free(dev);
        return NULL;
    }
    freeaddrinfo(res);

    if (send_msg(dev->sock_fd, CML_REMOTE_OP_PING, NULL, 0) != 0) {
        LOG_ERROR("remote_connect: ping send failed");
        close(dev->sock_fd);
        free(dev);
        return NULL;
    }

    uint32_t status;
    uint64_t sid = 0;
    uint32_t psize = 0;
    if (recv_resp(dev->sock_fd, &status, &sid, &psize) != 0 || status != CML_REMOTE_STATUS_OK) {
        LOG_ERROR("remote_connect: ping response failed");
        close(dev->sock_fd);
        free(dev);
        return NULL;
    }

    dev->session_id = sid;
    dev->connected = true;
    LOG_INFO("Connected to remote device %s:%d (session 0x%llx)",
             host, port, (unsigned long long)dev->session_id);
    return dev;
}

void cml_remote_disconnect(CMLRemoteDevice* dev) {
    if (!dev) return;
    if (dev->sock_fd >= 0) {
        close(dev->sock_fd);
        dev->sock_fd = -1;
    }
    dev->connected = false;
    free(dev);
}

bool cml_remote_is_connected(CMLRemoteDevice* dev) {
    return dev && dev->connected && dev->sock_fd >= 0;
}

uint64_t cml_remote_alloc(CMLRemoteDevice* dev, size_t size) {
    if (!cml_remote_is_connected(dev)) return 0;

    uint64_t net_size = (uint64_t)size;
    if (send_msg(dev->sock_fd, CML_REMOTE_OP_ALLOC, &net_size, sizeof(net_size)) != 0)
        return 0;

    uint32_t status;
    uint64_t handle = 0;
    uint32_t psize = 0;
    if (recv_resp(dev->sock_fd, &status, &handle, &psize) != 0)
        return 0;

    if (status != CML_REMOTE_STATUS_OK) return 0;
    return handle;
}

void cml_remote_free(CMLRemoteDevice* dev, uint64_t handle) {
    if (!cml_remote_is_connected(dev)) return;

    if (send_msg(dev->sock_fd, CML_REMOTE_OP_FREE, &handle, sizeof(handle)) != 0)
        return;

    uint32_t status, psize = 0;
    recv_resp(dev->sock_fd, &status, NULL, &psize);
}

int cml_remote_upload(CMLRemoteDevice* dev, uint64_t handle, const void* data, size_t n) {
    if (!cml_remote_is_connected(dev) || !data) return -1;

    size_t hdr_size = sizeof(handle) + sizeof(uint64_t);
    size_t total = hdr_size + n;
    uint8_t* payload = (uint8_t*)malloc(total);
    if (!payload) return -1;

    memcpy(payload, &handle, sizeof(handle));
    uint64_t data_len = (uint64_t)n;
    memcpy(payload + sizeof(handle), &data_len, sizeof(data_len));
    memcpy(payload + hdr_size, data, n);

    int ret = send_msg(dev->sock_fd, CML_REMOTE_OP_UPLOAD, payload, (uint32_t)total);
    free(payload);
    if (ret != 0) return -1;

    uint32_t status, psize = 0;
    if (recv_resp(dev->sock_fd, &status, NULL, &psize) != 0) return -1;
    return (status == CML_REMOTE_STATUS_OK) ? 0 : -1;
}

int cml_remote_download(CMLRemoteDevice* dev, uint64_t handle, void* data, size_t n) {
    if (!cml_remote_is_connected(dev) || !data) return -1;

    uint8_t payload[16];
    memcpy(payload, &handle, sizeof(handle));
    uint64_t data_len = (uint64_t)n;
    memcpy(payload + sizeof(handle), &data_len, sizeof(data_len));

    if (send_msg(dev->sock_fd, CML_REMOTE_OP_DOWNLOAD, payload, sizeof(payload)) != 0)
        return -1;

    uint32_t status;
    RespHeader rhdr;
    if (recv_all(dev->sock_fd, &rhdr, sizeof(rhdr)) != 0) return -1;
    status = ntohl(rhdr.status);
    uint32_t resp_size = ntohl(rhdr.payload_size);

    if (status != CML_REMOTE_STATUS_OK) return -1;

    size_t to_read = (resp_size < n) ? resp_size : n;
    if (recv_all(dev->sock_fd, data, to_read) != 0) return -1;

    if (resp_size > to_read) {
        uint8_t discard[4096];
        size_t remaining = resp_size - to_read;
        while (remaining > 0) {
            size_t chunk = remaining < sizeof(discard) ? remaining : sizeof(discard);
            if (recv_all(dev->sock_fd, discard, chunk) != 0) return -1;
            remaining -= chunk;
        }
    }

    return 0;
}

int cml_remote_execute(CMLRemoteDevice* dev, const char* kernel_source,
                       uint64_t* buffer_handles, int num_buffers,
                       uint32_t grid[3], uint32_t block[3]) {
    if (!cml_remote_is_connected(dev) || !kernel_source) return -1;

    uint32_t src_len = (uint32_t)strlen(kernel_source) + 1;
    uint32_t num_buf = (uint32_t)num_buffers;
    size_t handles_size = (size_t)num_buffers * sizeof(uint64_t);

    /* payload: [src_len(4)][source][num_buf(4)][handles...][grid(12)][block(12)] */
    size_t total = sizeof(uint32_t) + src_len + sizeof(uint32_t) + handles_size + 24;
    uint8_t* payload = (uint8_t*)malloc(total);
    if (!payload) return -1;

    size_t off = 0;
    memcpy(payload + off, &src_len, 4); off += 4;
    memcpy(payload + off, kernel_source, src_len); off += src_len;
    memcpy(payload + off, &num_buf, 4); off += 4;
    if (num_buffers > 0 && buffer_handles)
        memcpy(payload + off, buffer_handles, handles_size);
    off += handles_size;
    memcpy(payload + off, grid, 12); off += 12;
    memcpy(payload + off, block, 12);

    int ret = send_msg(dev->sock_fd, CML_REMOTE_OP_EXECUTE, payload, (uint32_t)total);
    free(payload);
    if (ret != 0) return -1;

    uint32_t status, psize = 0;
    if (recv_resp(dev->sock_fd, &status, NULL, &psize) != 0) return -1;
    return (status == CML_REMOTE_STATUS_OK) ? 0 : -1;
}

/* Server: allocation table */

#define MAX_ALLOCS 4096

typedef struct {
    uint64_t handle;
    void* ptr;
    size_t size;
} AllocEntry;

static AllocEntry g_allocs[MAX_ALLOCS];
static int g_num_allocs = 0;
static uint64_t g_next_handle = 1;

static uint64_t server_alloc(size_t size) {
    if (g_num_allocs >= MAX_ALLOCS) return 0;

    DeviceType best = device_get_best_available();
    void* ptr = device_alloc(size, best);
    if (!ptr) {
        ptr = malloc(size);
        if (!ptr) return 0;
    }

    uint64_t h = g_next_handle++;
    g_allocs[g_num_allocs].handle = h;
    g_allocs[g_num_allocs].ptr = ptr;
    g_allocs[g_num_allocs].size = size;
    g_num_allocs++;
    return h;
}

static AllocEntry* server_find(uint64_t handle) {
    for (int i = 0; i < g_num_allocs; i++) {
        if (g_allocs[i].handle == handle) return &g_allocs[i];
    }
    return NULL;
}

static void server_free_handle(uint64_t handle) {
    for (int i = 0; i < g_num_allocs; i++) {
        if (g_allocs[i].handle == handle) {
            free(g_allocs[i].ptr);
            g_allocs[i] = g_allocs[g_num_allocs - 1];
            g_num_allocs--;
            return;
        }
    }
}

static void server_free_all(void) {
    for (int i = 0; i < g_num_allocs; i++) {
        free(g_allocs[i].ptr);
    }
    g_num_allocs = 0;
}

/* Server: handle one client */

static void handle_client(int client_fd) {
    while (1) {
        MsgHeader hdr;
        if (recv_all(client_fd, &hdr, sizeof(hdr)) != 0) break;

        uint32_t opcode = ntohl(hdr.opcode);
        uint32_t psize = ntohl(hdr.payload_size);

        uint8_t* payload = NULL;
        if (psize > 0) {
            payload = (uint8_t*)malloc(psize);
            if (!payload) break;
            if (recv_all(client_fd, payload, psize) != 0) {
                free(payload);
                break;
            }
        }

        switch (opcode) {
        case CML_REMOTE_OP_PING: {
            uint64_t sid = (uint64_t)time(NULL) ^ (uint64_t)client_fd;
            send_resp(client_fd, CML_REMOTE_STATUS_OK, &sid, sizeof(sid));
            break;
        }
        case CML_REMOTE_OP_ALLOC: {
            uint64_t req_size = 0;
            if (psize >= sizeof(uint64_t))
                memcpy(&req_size, payload, sizeof(uint64_t));
            uint64_t h = server_alloc((size_t)req_size);
            uint32_t st = (h != 0) ? CML_REMOTE_STATUS_OK : CML_REMOTE_STATUS_ERROR;
            send_resp(client_fd, st, &h, sizeof(h));
            break;
        }
        case CML_REMOTE_OP_FREE: {
            uint64_t h = 0;
            if (psize >= sizeof(uint64_t))
                memcpy(&h, payload, sizeof(uint64_t));
            server_free_handle(h);
            send_resp(client_fd, CML_REMOTE_STATUS_OK, NULL, 0);
            break;
        }
        case CML_REMOTE_OP_UPLOAD: {
            if (psize < 16) {
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            uint64_t h, data_len;
            memcpy(&h, payload, sizeof(h));
            memcpy(&data_len, payload + sizeof(h), sizeof(data_len));
            AllocEntry* entry = server_find(h);
            if (!entry || data_len > entry->size) {
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            size_t copy_len = (data_len < entry->size) ? (size_t)data_len : entry->size;
            memcpy(entry->ptr, payload + 16, copy_len);
            send_resp(client_fd, CML_REMOTE_STATUS_OK, NULL, 0);
            break;
        }
        case CML_REMOTE_OP_DOWNLOAD: {
            if (psize < 16) {
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            uint64_t h, data_len;
            memcpy(&h, payload, sizeof(h));
            memcpy(&data_len, payload + sizeof(h), sizeof(data_len));
            AllocEntry* entry = server_find(h);
            if (!entry) {
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            size_t send_len = (data_len < entry->size) ? (size_t)data_len : entry->size;
            send_resp(client_fd, CML_REMOTE_STATUS_OK, entry->ptr, (uint32_t)send_len);
            break;
        }
        case CML_REMOTE_OP_EXECUTE: {
#ifdef CML_HAS_DLFCN
            /* Parse payload:
             * [src_len:uint32][kernel_source:src_len bytes]
             * [num_buf:uint32][handles:num_buf*uint64]
             * [grid:3*uint32][block:3*uint32]
             */
            if (psize < 8) {
                LOG_ERROR("remote_server: execute payload too small (%u bytes)", psize);
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            size_t off2 = 0;
            uint32_t src_len2 = 0;
            memcpy(&src_len2, payload + off2, 4); off2 += 4;
            if (src_len2 == 0 || off2 + src_len2 > psize) {
                LOG_ERROR("remote_server: execute bad src_len %u", src_len2);
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            char* kernel_src = (char*)(payload + off2); off2 += src_len2;
            /* ensure NUL-terminated within payload */
            kernel_src[src_len2 - 1] = '\0';

            if (off2 + 4 > psize) {
                LOG_ERROR("remote_server: execute payload truncated at num_buf");
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            uint32_t num_buf2 = 0;
            memcpy(&num_buf2, payload + off2, 4); off2 += 4;

            size_t handles_bytes = (size_t)num_buf2 * sizeof(uint64_t);
            if (off2 + handles_bytes + 24 > psize) {
                LOG_ERROR("remote_server: execute payload truncated at handles/grid/block");
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            uint64_t* handles2 = (uint64_t*)(payload + off2); off2 += handles_bytes;
            uint32_t grid2[3], block2[3];
            memcpy(grid2,  payload + off2, 12); off2 += 12;
            memcpy(block2, payload + off2, 12);

            /* Resolve handles to buffer pointers */
            float** bufs = NULL;
            if (num_buf2 > 0) {
                bufs = (float**)malloc((size_t)num_buf2 * sizeof(float*));
                if (!bufs) {
                    LOG_ERROR("remote_server: OOM resolving buffer handles");
                    send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                    break;
                }
                int lookup_ok = 1;
                for (uint32_t bi = 0; bi < num_buf2; bi++) {
                    AllocEntry* ae = server_find(handles2[bi]);
                    if (!ae) {
                        LOG_ERROR("remote_server: unknown handle 0x%llx",
                                  (unsigned long long)handles2[bi]);
                        lookup_ok = 0;
                        break;
                    }
                    bufs[bi] = (float*)ae->ptr;
                }
                if (!lookup_ok) {
                    free(bufs);
                    send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                    break;
                }
            }

            /* Write kernel source to a temp file, compile with gcc/cc */
            char src_path[] = "/tmp/cml_kernel_XXXXXX.c";
            char so_path[]  = "/tmp/cml_kernel_XXXXXX.so";

            /* Use mkstemp for the .c file, then build .so path from it */
            int src_fd2 = mkstemps(src_path, 2); /* suffix len = 2 for ".c" */
            if (src_fd2 < 0) {
                LOG_ERROR("remote_server: mkstemps failed: %s", strerror(errno));
                free(bufs);
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }
            /* Derive .so path by replacing the last 2 chars (.c) with .so */
            memcpy(so_path, src_path, sizeof(src_path));
            size_t slen = strlen(so_path);
            so_path[slen - 2] = '.';
            so_path[slen - 1] = 's';
            /* need one more char; so_path has room — adjust: use a local buf */
            char so_path2[64];
            snprintf(so_path2, sizeof(so_path2), "%.*s.so", (int)(slen - 2), so_path);

            /* Write kernel source */
            size_t src_written = 0;
            size_t src_total = (size_t)(src_len2 - 1); /* exclude NUL */
            while (src_written < src_total) {
                ssize_t nw = write(src_fd2, kernel_src + src_written, src_total - src_written);
                if (nw <= 0) break;
                src_written += (size_t)nw;
            }
            close(src_fd2);

            /* Compile */
            char compile_cmd[512];
            snprintf(compile_cmd, sizeof(compile_cmd),
                     "gcc -O2 -shared -fPIC -o %s %s 2>/dev/null || "
                     "cc  -O2 -shared -fPIC -o %s %s 2>/dev/null",
                     so_path2, src_path,
                     so_path2, src_path);
            int compile_ret = system(compile_cmd);
            unlink(src_path);

            if (compile_ret != 0) {
                LOG_ERROR("remote_server: kernel compilation failed (exit %d)", compile_ret);
                unlink(so_path2);
                free(bufs);
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }

            /* Load and call the kernel */
            void* dl = dlopen(so_path2, RTLD_NOW | RTLD_LOCAL);
            if (!dl) {
                LOG_ERROR("remote_server: dlopen failed: %s", dlerror());
                unlink(so_path2);
                free(bufs);
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }

            typedef void (*kernel_fn_t)(float** bufs, int num_bufs, uint32_t* grid, uint32_t* block);
            kernel_fn_t kfn = (kernel_fn_t)dlsym(dl, "cml_kernel");
            if (!kfn) {
                LOG_ERROR("remote_server: dlsym(cml_kernel) failed: %s", dlerror());
                dlclose(dl);
                unlink(so_path2);
                free(bufs);
                send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
                break;
            }

            kfn(bufs, (int)num_buf2, grid2, block2);

            dlclose(dl);
            unlink(so_path2);
            free(bufs);

            LOG_INFO("remote_server: kernel executed successfully (%u bufs)", num_buf2);
            send_resp(client_fd, CML_REMOTE_STATUS_OK, NULL, 0);
#else
            /* No dlopen support: no-op stub */
            LOG_INFO("remote_server: execute request (no-op: dlfcn unavailable)");
            send_resp(client_fd, CML_REMOTE_STATUS_OK, NULL, 0);
#endif
            break;
        }
        default:
            send_resp(client_fd, CML_REMOTE_STATUS_ERROR, NULL, 0);
            break;
        }

        free(payload);
    }
}

/* Server lifecycle */

CMLRemoteServer* cml_remote_server_create(int port) {
    if (port <= 0) return NULL;

    CMLRemoteServer* srv = (CMLRemoteServer*)calloc(1, sizeof(CMLRemoteServer));
    if (!srv) return NULL;

    srv->port = port;
    srv->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (srv->listen_fd < 0) {
        LOG_ERROR("remote_server: socket() failed: %s", strerror(errno));
        free(srv);
        return NULL;
    }

    int opt = 1;
    setsockopt(srv->listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons((uint16_t)port);

    if (bind(srv->listen_fd, (struct sockaddr*)&addr, sizeof(addr)) != 0) {
        LOG_ERROR("remote_server: bind() port %d failed: %s", port, strerror(errno));
        close(srv->listen_fd);
        free(srv);
        return NULL;
    }

    if (listen(srv->listen_fd, 8) != 0) {
        LOG_ERROR("remote_server: listen() failed: %s", strerror(errno));
        close(srv->listen_fd);
        free(srv);
        return NULL;
    }

    LOG_INFO("Remote server created on port %d", port);
    return srv;
}

int cml_remote_server_run(CMLRemoteServer* srv) {
    if (!srv) return -1;
    srv->running = true;

    LOG_INFO("Remote server listening on port %d", srv->port);

    while (srv->running) {
        struct sockaddr_in client_addr;
        socklen_t addr_len = sizeof(client_addr);
        int client_fd = accept(srv->listen_fd, (struct sockaddr*)&client_addr, &addr_len);
        if (client_fd < 0) {
            if (!srv->running) break;
            if (errno == EINTR) continue;
            LOG_ERROR("remote_server: accept() failed: %s", strerror(errno));
            continue;
        }

        int flag = 1;
        setsockopt(client_fd, IPPROTO_TCP, TCP_NODELAY, &flag, sizeof(flag));

        char addr_str[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &client_addr.sin_addr, addr_str, sizeof(addr_str));
        LOG_INFO("Remote server: client connected from %s:%d",
                 addr_str, ntohs(client_addr.sin_port));

        handle_client(client_fd);

        close(client_fd);
        LOG_INFO("Remote server: client disconnected");
    }

    return 0;
}

void cml_remote_server_stop(CMLRemoteServer* srv) {
    if (!srv) return;
    srv->running = false;
    if (srv->listen_fd >= 0) {
        shutdown(srv->listen_fd, SHUT_RDWR);
    }
}

void cml_remote_server_free(CMLRemoteServer* srv) {
    if (!srv) return;
    cml_remote_server_stop(srv);
    if (srv->listen_fd >= 0) {
        close(srv->listen_fd);
    }
    server_free_all();
    free(srv);
}
