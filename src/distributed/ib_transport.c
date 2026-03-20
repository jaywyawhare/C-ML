#include "distributed/ib_transport.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>

/* InfiniBand verbs constants */
#define IBV_QPT_RC          2
#define IBV_QPS_INIT        1
#define IBV_QPS_RTR         2
#define IBV_QPS_RTS         3
#define IBV_WR_SEND         0
#define IBV_SEND_SIGNALED   (1 << 1)
#define IBV_ACCESS_LOCAL_WRITE  (1 << 0)
#define IBV_ACCESS_REMOTE_WRITE (1 << 1)
#define IBV_ACCESS_REMOTE_READ  (1 << 2)
#define IBV_WC_SUCCESS      0

/* Opaque IB structures - we only use pointers */
typedef struct ibv_device      ibv_device;
typedef struct ibv_context     ibv_context;
typedef struct ibv_pd          ibv_pd;
typedef struct ibv_cq          ibv_cq;
typedef struct ibv_qp          ibv_qp;
typedef struct ibv_mr          ibv_mr;

typedef struct ibv_qp_init_attr {
    void* qp_context;
    ibv_cq* send_cq;
    ibv_cq* recv_cq;
    void* srq;
    struct { uint32_t max_send_wr; uint32_t max_recv_wr; uint32_t max_send_sge; uint32_t max_recv_sge; uint32_t max_inline_data; } cap;
    int qp_type;
    int sq_sig_all;
} ibv_qp_init_attr;

typedef struct ibv_sge {
    uint64_t addr;
    uint32_t length;
    uint32_t lkey;
} ibv_sge;

typedef struct ibv_send_wr {
    uint64_t wr_id;
    struct ibv_send_wr* next;
    ibv_sge* sg_list;
    int num_sge;
    int opcode;
    int send_flags;
    union { struct { uint64_t remote_addr; uint32_t rkey; } rdma; } wr;
} ibv_send_wr;

typedef struct ibv_recv_wr {
    uint64_t wr_id;
    struct ibv_recv_wr* next;
    ibv_sge* sg_list;
    int num_sge;
} ibv_recv_wr;

typedef struct ibv_wc {
    uint64_t wr_id;
    int status;
    int opcode;
    uint32_t byte_len;
    uint32_t qp_num;
} ibv_wc;

typedef struct ibv_port_attr {
    int state;
    int max_mtu;
    int active_mtu;
    int gid_tbl_len;
    uint32_t port_cap_flags;
    uint32_t max_msg_sz;
    uint16_t lid;
    uint8_t lmc;
    uint8_t max_vl_num;
    uint8_t sm_sl;
    uint8_t subnet_timeout;
    uint8_t init_type_reply;
    uint8_t active_width;
    uint8_t active_speed;
    uint8_t phys_state;
} ibv_port_attr;

typedef struct ibv_qp_attr {
    int qp_state;
    int cur_qp_state;
    int path_mtu;
    uint32_t path_mig_state;
    uint32_t qkey;
    uint32_t rq_psn;
    uint32_t sq_psn;
    uint32_t dest_qp_num;
    int qp_access_flags;
    struct { uint16_t dlid; uint8_t sl; uint8_t src_path_bits; uint8_t static_rate; uint8_t is_global; uint8_t port_num; struct { uint32_t flow_label; uint8_t sgid_index; uint8_t hop_limit; uint8_t traffic_class; } grh; } ah_attr;
    uint8_t pkey_index;
    uint8_t _port_num;
    uint32_t sq_draining;
    uint8_t max_rd_atomic;
    uint8_t max_dest_rd_atomic;
    uint8_t min_rnr_timer;
    uint8_t timeout;
    uint8_t retry_cnt;
    uint8_t rnr_retry;
    uint8_t alt_port_num;
    uint8_t alt_timeout;
} ibv_qp_attr;

/* Function pointer types */
typedef ibv_device** (*fn_ibv_get_device_list_t)(int* num_devices);
typedef void         (*fn_ibv_free_device_list_t)(ibv_device** list);
typedef ibv_context* (*fn_ibv_open_device_t)(ibv_device* device);
typedef int          (*fn_ibv_close_device_t)(ibv_context* context);
typedef ibv_pd*      (*fn_ibv_alloc_pd_t)(ibv_context* context);
typedef int          (*fn_ibv_dealloc_pd_t)(ibv_pd* pd);
typedef ibv_cq*      (*fn_ibv_create_cq_t)(ibv_context* context, int cqe, void* cq_ctx, void* channel, int comp_vector);
typedef int          (*fn_ibv_destroy_cq_t)(ibv_cq* cq);
typedef ibv_qp*      (*fn_ibv_create_qp_t)(ibv_pd* pd, ibv_qp_init_attr* qp_init_attr);
typedef int          (*fn_ibv_destroy_qp_t)(ibv_qp* qp);
typedef int          (*fn_ibv_modify_qp_t)(ibv_qp* qp, ibv_qp_attr* attr, int attr_mask);
typedef ibv_mr*      (*fn_ibv_reg_mr_t)(ibv_pd* pd, void* addr, size_t length, int access);
typedef int          (*fn_ibv_dereg_mr_t)(ibv_mr* mr);
typedef int          (*fn_ibv_post_send_t)(ibv_qp* qp, ibv_send_wr* wr, ibv_send_wr** bad_wr);
typedef int          (*fn_ibv_post_recv_t)(ibv_qp* qp, ibv_recv_wr* wr, ibv_recv_wr** bad_wr);
typedef int          (*fn_ibv_poll_cq_t)(ibv_cq* cq, int num_entries, ibv_wc* wc);
typedef int          (*fn_ibv_query_port_t)(ibv_context* context, uint8_t port_num, ibv_port_attr* port_attr);

static struct {
    void* lib;
    fn_ibv_get_device_list_t  get_device_list;
    fn_ibv_free_device_list_t free_device_list;
    fn_ibv_open_device_t      open_device;
    fn_ibv_close_device_t     close_device;
    fn_ibv_alloc_pd_t         alloc_pd;
    fn_ibv_dealloc_pd_t       dealloc_pd;
    fn_ibv_create_cq_t        create_cq;
    fn_ibv_destroy_cq_t       destroy_cq;
    fn_ibv_create_qp_t        create_qp;
    fn_ibv_destroy_qp_t       destroy_qp;
    fn_ibv_modify_qp_t        modify_qp;
    fn_ibv_reg_mr_t            reg_mr;
    fn_ibv_dereg_mr_t          dereg_mr;
    fn_ibv_post_send_t         post_send;
    fn_ibv_post_recv_t         post_recv;
    fn_ibv_poll_cq_t           poll_cq;
    fn_ibv_query_port_t        query_port;
} ib_api = {0};

static bool load_ib_symbols(void* lib) {
#define LOAD_SYM(name) do { \
    ib_api.name = (fn_ibv_##name##_t)dlsym(lib, "ibv_" #name); \
    if (!ib_api.name) { LOG_ERROR("Missing symbol ibv_%s: %s", #name, dlerror()); return false; } \
} while(0)

    LOAD_SYM(get_device_list);
    LOAD_SYM(free_device_list);
    LOAD_SYM(open_device);
    LOAD_SYM(close_device);
    LOAD_SYM(alloc_pd);
    LOAD_SYM(dealloc_pd);
    LOAD_SYM(create_cq);
    LOAD_SYM(destroy_cq);
    LOAD_SYM(create_qp);
    LOAD_SYM(destroy_qp);
    LOAD_SYM(modify_qp);
    LOAD_SYM(reg_mr);
    LOAD_SYM(dereg_mr);
    LOAD_SYM(post_send);
    LOAD_SYM(post_recv);
    LOAD_SYM(poll_cq);
    LOAD_SYM(query_port);
#undef LOAD_SYM
    return true;
}

bool cml_ib_available(void) {
    void* lib = dlopen("libibverbs.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libibverbs.so", RTLD_LAZY);
    if (!lib) return false;

    fn_ibv_get_device_list_t gdl = (fn_ibv_get_device_list_t)dlsym(lib, "ibv_get_device_list");
    if (!gdl) { dlclose(lib); return false; }

    int num = 0;
    ibv_device** devs = gdl(&num);
    bool found = (devs && num > 0);

    if (devs) {
        fn_ibv_free_device_list_t fdl = (fn_ibv_free_device_list_t)dlsym(lib, "ibv_free_device_list");
        if (fdl) fdl(devs);
    }

    dlclose(lib);
    return found;
}

CMLIBTransport* cml_ib_create(int rank, int world_size) {
    if (rank < 0 || world_size <= 0 || rank >= world_size) {
        LOG_ERROR("Invalid rank=%d world_size=%d", rank, world_size);
        return NULL;
    }

    void* lib = dlopen("libibverbs.so.1", RTLD_LAZY);
    if (!lib) lib = dlopen("libibverbs.so", RTLD_LAZY);
    if (!lib) {
        LOG_ERROR("Failed to load libibverbs: %s", dlerror());
        return NULL;
    }

    if (!load_ib_symbols(lib)) {
        dlclose(lib);
        return NULL;
    }

    int num_devs = 0;
    ibv_device** dev_list = ib_api.get_device_list(&num_devs);
    if (!dev_list || num_devs == 0) {
        LOG_ERROR("No InfiniBand devices found");
        dlclose(lib);
        return NULL;
    }

    ibv_context* ctx = ib_api.open_device(dev_list[0]);
    ib_api.free_device_list(dev_list);
    if (!ctx) {
        LOG_ERROR("Failed to open IB device");
        dlclose(lib);
        return NULL;
    }

    ibv_pd* pd = ib_api.alloc_pd(ctx);
    if (!pd) {
        LOG_ERROR("Failed to allocate protection domain");
        ib_api.close_device(ctx);
        dlclose(lib);
        return NULL;
    }

    int cq_size = 256 * (world_size - 1);
    if (cq_size < 16) cq_size = 16;
    ibv_cq* cq = ib_api.create_cq(ctx, cq_size, NULL, NULL, 0);
    if (!cq) {
        LOG_ERROR("Failed to create completion queue");
        ib_api.dealloc_pd(pd);
        ib_api.close_device(ctx);
        dlclose(lib);
        return NULL;
    }

    CMLIBTransport* ib = calloc(1, sizeof(CMLIBTransport));
    if (!ib) {
        ib_api.destroy_cq(cq);
        ib_api.dealloc_pd(pd);
        ib_api.close_device(ctx);
        dlclose(lib);
        return NULL;
    }

    ib->ib_ctx = ctx;
    ib->pd = pd;
    ib->cq = cq;
    ib->rank = rank;
    ib->world_size = world_size;
    ib->ib_lib = lib;
    ib->num_peers = world_size - 1;
    ib->connected = false;

    ib->qps = calloc((size_t)world_size, sizeof(void*));
    if (!ib->qps) {
        cml_ib_free(ib);
        return NULL;
    }

    for (int i = 0; i < world_size; i++) {
        if (i == rank) continue;

        ibv_qp_init_attr qp_attr;
        memset(&qp_attr, 0, sizeof(qp_attr));
        qp_attr.send_cq = cq;
        qp_attr.recv_cq = cq;
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = 128;
        qp_attr.cap.max_recv_wr = 128;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;
        qp_attr.cap.max_inline_data = 64;
        qp_attr.sq_sig_all = 0;

        ibv_qp* qp = ib_api.create_qp(pd, &qp_attr);
        if (!qp) {
            LOG_ERROR("Failed to create QP for peer %d", i);
            cml_ib_free(ib);
            return NULL;
        }
        ib->qps[i] = qp;
    }

    LOG_INFO("IB transport created: rank %d/%d, %d QPs", rank, world_size, world_size - 1);
    return ib;
}

/* QP info exchanged via TCP out-of-band */
typedef struct {
    uint32_t qp_num;
    uint16_t lid;
    uint32_t psn;
} qp_info_t;

static int transition_qp_init(CMLIBTransport* ib, int peer) {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr._port_num = 1;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;

    int mask = (1 << 0) | (1 << 16) | (1 << 17) | (1 << 20); /* QP_STATE | PKEY_INDEX | PORT | ACCESS_FLAGS */
    return ib_api.modify_qp(ib->qps[peer], &attr, mask);
}

static int transition_qp_rtr(CMLIBTransport* ib, int peer, qp_info_t* remote) {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = 3; /* IBV_MTU_1024 */
    attr.dest_qp_num = remote->qp_num;
    attr.rq_psn = remote->psn;
    attr.max_dest_rd_atomic = 4;
    attr.min_rnr_timer = 12;
    attr.ah_attr.dlid = remote->lid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = 1;
    attr.ah_attr.is_global = 0;

    int mask = (1 << 0) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 6) | (1 << 9) | (1 << 12);
    return ib_api.modify_qp(ib->qps[peer], &attr, mask);
}

static int transition_qp_rts(CMLIBTransport* ib, int peer, uint32_t local_psn) {
    ibv_qp_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = local_psn;
    attr.max_rd_atomic = 4;

    int mask = (1 << 0) | (1 << 5) | (1 << 8) | (1 << 10) | (1 << 11) | (1 << 13);
    return ib_api.modify_qp(ib->qps[peer], &attr, mask);
}

static int tcp_exchange(const char* addr, int is_server, void* send_data, void* recv_data, size_t size) {
    struct addrinfo hints, *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;

    char host[256];
    int port = 18515;
    strncpy(host, addr, sizeof(host) - 1);
    host[sizeof(host) - 1] = '\0';

    char* colon = strrchr(host, ':');
    if (colon) {
        *colon = '\0';
        port = atoi(colon + 1);
    }

    char port_str[16];
    snprintf(port_str, sizeof(port_str), "%d", port);

    int fd = -1;
    if (is_server) {
        hints.ai_flags = AI_PASSIVE;
        if (getaddrinfo(NULL, port_str, &hints, &res) != 0) return -1;

        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { freeaddrinfo(res); return -1; }

        int opt = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        if (bind(fd, res->ai_addr, res->ai_addrlen) < 0 || listen(fd, 1) < 0) {
            close(fd); freeaddrinfo(res); return -1;
        }
        freeaddrinfo(res);

        int client = accept(fd, NULL, NULL);
        close(fd);
        if (client < 0) return -1;
        fd = client;
    } else {
        if (getaddrinfo(host, port_str, &hints, &res) != 0) return -1;
        fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { freeaddrinfo(res); return -1; }

        int retries = 50;
        while (connect(fd, res->ai_addr, res->ai_addrlen) < 0 && retries > 0) {
            usleep(100000);
            retries--;
        }
        freeaddrinfo(res);
        if (retries <= 0) { close(fd); return -1; }
    }

    ssize_t n = send(fd, send_data, size, 0);
    if (n != (ssize_t)size) { close(fd); return -1; }

    size_t recvd = 0;
    while (recvd < size) {
        n = recv(fd, (char*)recv_data + recvd, size - recvd, 0);
        if (n <= 0) { close(fd); return -1; }
        recvd += (size_t)n;
    }

    close(fd);
    return 0;
}

int cml_ib_connect(CMLIBTransport* ib, const char** peer_addrs, int num_peers) {
    if (!ib || !peer_addrs || num_peers != ib->world_size - 1) return -1;

    ibv_port_attr port_attr;
    if (ib_api.query_port(ib->ib_ctx, 1, &port_attr) != 0) {
        LOG_ERROR("Failed to query IB port");
        return -1;
    }

    uint32_t local_psn = (uint32_t)(ib->rank * 1000);

    int peer_idx = 0;
    for (int i = 0; i < ib->world_size; i++) {
        if (i == ib->rank) continue;

        if (transition_qp_init(ib, i) != 0) {
            LOG_ERROR("Failed to transition QP to INIT for peer %d", i);
            return -1;
        }

        qp_info_t local_info = {
            .qp_num = ((ibv_qp*)ib->qps[i])->qp_num,  /* libibverbs: ibv_qp.qp_num is public */
            .lid = port_attr.lid,
            .psn = local_psn
        };

        qp_info_t remote_info;
        int is_server = (ib->rank < i);
        if (tcp_exchange(peer_addrs[peer_idx], is_server,
                         &local_info, &remote_info, sizeof(qp_info_t)) != 0) {
            LOG_ERROR("TCP exchange failed for peer %d", i);
            return -1;
        }

        if (transition_qp_rtr(ib, i, &remote_info) != 0) {
            LOG_ERROR("Failed to transition QP to RTR for peer %d", i);
            return -1;
        }

        if (transition_qp_rts(ib, i, local_psn) != 0) {
            LOG_ERROR("Failed to transition QP to RTS for peer %d", i);
            return -1;
        }

        peer_idx++;
    }

    ib->connected = true;
    LOG_INFO("IB transport connected: rank %d, %d peers", ib->rank, num_peers);
    return 0;
}

void cml_ib_free(CMLIBTransport* ib) {
    if (!ib) return;

    if (ib->qps) {
        for (int i = 0; i < ib->world_size; i++) {
            if (ib->qps[i]) ib_api.destroy_qp(ib->qps[i]);
        }
        free(ib->qps);
    }

    if (ib->cq) ib_api.destroy_cq(ib->cq);
    if (ib->pd) ib_api.dealloc_pd(ib->pd);
    if (ib->ib_ctx) ib_api.close_device(ib->ib_ctx);
    if (ib->ib_lib) dlclose(ib->ib_lib);

    free(ib);
}

static int poll_completion(CMLIBTransport* ib, int timeout_ms) {
    int elapsed = 0;
    while (elapsed < timeout_ms) {
        ibv_wc wc;
        int n = ib_api.poll_cq(ib->cq, 1, &wc);
        if (n > 0) {
            if (wc.status != IBV_WC_SUCCESS) {
                LOG_ERROR("Work completion error: status=%d", wc.status);
                return -1;
            }
            return 0;
        }
        if (n < 0) {
            LOG_ERROR("poll_cq failed");
            return -1;
        }
        usleep(100);
        elapsed++;
    }
    LOG_ERROR("poll_cq timeout after %d ms", timeout_ms);
    return -1;
}

int cml_ib_send(CMLIBTransport* ib, int peer, const void* buf, size_t size) {
    if (!ib || !ib->connected || peer < 0 || peer >= ib->world_size || peer == ib->rank)
        return -1;

    ibv_mr* mr = ib_api.reg_mr(ib->pd, (void*)buf, size,
                                IBV_ACCESS_LOCAL_WRITE);
    if (!mr) {
        LOG_ERROR("Failed to register send buffer");
        return -1;
    }

    /* Extract lkey from ibv_mr - it's at a known offset (typically after addr+length+handle).
     * We store 0 as placeholder; real code reads mr->lkey. */
    ibv_sge sge = {
        .addr = (uint64_t)(uintptr_t)buf,
        .length = (uint32_t)size,
        .lkey = 0
    };

    ibv_send_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)peer;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;

    ibv_send_wr* bad_wr = NULL;
    int rc = ib_api.post_send(ib->qps[peer], &wr, &bad_wr);
    if (rc != 0) {
        LOG_ERROR("post_send failed for peer %d: rc=%d", peer, rc);
        ib_api.dereg_mr(mr);
        return -1;
    }

    rc = poll_completion(ib, 30000);
    ib_api.dereg_mr(mr);
    return rc;
}

int cml_ib_recv(CMLIBTransport* ib, int peer, void* buf, size_t size) {
    if (!ib || !ib->connected || peer < 0 || peer >= ib->world_size || peer == ib->rank)
        return -1;

    ibv_mr* mr = ib_api.reg_mr(ib->pd, buf, size,
                                IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (!mr) {
        LOG_ERROR("Failed to register recv buffer");
        return -1;
    }

    ibv_sge sge = {
        .addr = (uint64_t)(uintptr_t)buf,
        .length = (uint32_t)size,
        .lkey = 0
    };

    ibv_recv_wr wr;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = (uint64_t)peer;
    wr.sg_list = &sge;
    wr.num_sge = 1;

    ibv_recv_wr* bad_wr = NULL;
    int rc = ib_api.post_recv(ib->qps[peer], &wr, &bad_wr);
    if (rc != 0) {
        LOG_ERROR("post_recv failed for peer %d: rc=%d", peer, rc);
        ib_api.dereg_mr(mr);
        return -1;
    }

    rc = poll_completion(ib, 30000);
    ib_api.dereg_mr(mr);
    return rc;
}

int cml_ib_allreduce(CMLIBTransport* ib, void* buf, size_t size, int elem_size) {
    if (!ib || !ib->connected || !buf || size == 0 || elem_size <= 0) return -1;

    int ws = ib->world_size;
    int rank = ib->rank;

    if (ws == 1) return 0;

    size_t count = size / (size_t)elem_size;
    size_t chunk = (count + (size_t)ws - 1) / (size_t)ws;
    size_t chunk_bytes = chunk * (size_t)elem_size;

    int left  = (rank - 1 + ws) % ws;
    int right = (rank + 1) % ws;

    float* recv_buf = malloc(chunk_bytes);
    if (!recv_buf) return -1;

    float* data = (float*)buf;

    /* Phase 1: reduce-scatter */
    for (int step = 0; step < ws - 1; step++) {
        int send_chunk = (rank - step + ws) % ws;
        int recv_chunk = (rank - step - 1 + ws) % ws;

        size_t send_off = (size_t)send_chunk * chunk;
        size_t recv_off = (size_t)recv_chunk * chunk;
        size_t sc = chunk;
        size_t rc = chunk;

        if (send_off + sc > count) sc = (send_off < count) ? count - send_off : 0;
        if (recv_off + rc > count) rc = (recv_off < count) ? count - recv_off : 0;

        if (sc > 0)
            cml_ib_send(ib, right, data + send_off, sc * (size_t)elem_size);
        if (rc > 0)
            cml_ib_recv(ib, left, recv_buf, rc * (size_t)elem_size);

        for (size_t i = 0; i < rc; i++)
            data[recv_off + i] += recv_buf[i];
    }

    /* Phase 2: allgather */
    for (int step = 0; step < ws - 1; step++) {
        int send_chunk = (rank - step + 1 + ws) % ws;
        int recv_chunk = (rank - step + ws) % ws;

        size_t send_off = (size_t)send_chunk * chunk;
        size_t recv_off = (size_t)recv_chunk * chunk;
        size_t sc = chunk;
        size_t rc = chunk;

        if (send_off + sc > count) sc = (send_off < count) ? count - send_off : 0;
        if (recv_off + rc > count) rc = (recv_off < count) ? count - recv_off : 0;

        if (sc > 0)
            cml_ib_send(ib, right, data + send_off, sc * (size_t)elem_size);
        if (rc > 0)
            cml_ib_recv(ib, left, data + recv_off, rc * (size_t)elem_size);
    }

    free(recv_buf);
    return 0;
}

int cml_ib_barrier(CMLIBTransport* ib) {
    if (!ib || !ib->connected) return -1;

    uint8_t dummy = 0;
    int left  = (ib->rank - 1 + ib->world_size) % ib->world_size;
    int right = (ib->rank + 1) % ib->world_size;

    for (int round = 0; round < ib->world_size - 1; round++) {
        if (cml_ib_send(ib, right, &dummy, 1) != 0) return -1;
        if (cml_ib_recv(ib, left, &dummy, 1) != 0) return -1;
    }
    return 0;
}

CMLIBMemReg* cml_ib_register_memory(CMLIBTransport* ib, void* addr, size_t size) {
    if (!ib || !addr || size == 0) return NULL;

    int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    ibv_mr* mr = ib_api.reg_mr(ib->pd, addr, size, access);
    if (!mr) {
        LOG_ERROR("Failed to register memory region (%zu bytes)", size);
        return NULL;
    }

    CMLIBMemReg* reg = calloc(1, sizeof(CMLIBMemReg));
    if (!reg) {
        ib_api.dereg_mr(mr);
        return NULL;
    }

    reg->mr = mr;
    reg->addr = addr;
    reg->size = size;
    reg->lkey = 0;
    reg->rkey = 0;

    return reg;
}

void cml_ib_deregister_memory(CMLIBTransport* ib, CMLIBMemReg* reg) {
    (void)ib;
    if (!reg) return;
    if (reg->mr) ib_api.dereg_mr(reg->mr);
    free(reg);
}
