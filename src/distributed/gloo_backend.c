#include "distributed/comm_backend.h"
#include "distributed/distributed.h"
#include "distributed/ring_allreduce.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>

#define GLOO_DEFAULT_PORT_BASE 29500
#define GLOO_DEFAULT_MASTER_ADDR "127.0.0.1"
#define GLOO_MAX_CONNECT_RETRIES 50
#define GLOO_CONNECT_RETRY_US 100000 /* 100ms */

typedef struct GlooContext {
    int rank;
    int world_size;
    int listen_fd;         /* Listening socket for this rank */
    int* peer_fds;         /* Array of connected socket fds, indexed by rank */
    int port_base;
    char master_addr[256];
} GlooContext;

/*
 * File-static context pointer. This is necessary because distributed.c
 * currently passes NULL as the ctx argument to all backend ops. Once
 * distributed.c is updated to propagate backend_ctx properly, this
 * static can be removed and ctx parameters used directly.
 */
static GlooContext* g_gloo_ctx = NULL;

static GlooContext* get_gloo_ctx(void* ctx) {
    return ctx ? (GlooContext*)ctx : g_gloo_ctx;
}

static int send_all(int fd, const void* buf, size_t len) {
    const char* p = (const char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = send(fd, p, remaining, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

static int recv_all(int fd, void* buf, size_t len) {
    char* p = (char*)buf;
    size_t remaining = len;
    while (remaining > 0) {
        ssize_t n = recv(fd, p, remaining, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return -1;
        }
        p += n;
        remaining -= (size_t)n;
    }
    return 0;
}

/* Forward declaration so async can call allreduce */
static int gloo_allreduce(Tensor* tensor, DistReduceOp op, void* ctx);

/* Protocol: send tag (int), then data size (size_t), then raw data bytes */
static int gloo_send(Tensor* tensor, int dst_rank, int tag, void* ctx) {
    GlooContext* gctx = get_gloo_ctx(ctx);
    if (!tensor || !tensor->data)
        return -1;

    if (!gctx || gctx->world_size <= 1)
        return 0;

    if (dst_rank < 0 || dst_rank >= gctx->world_size || dst_rank == gctx->rank) {
        LOG_ERROR("Gloo send: invalid dst_rank %d", dst_rank);
        return -1;
    }

    int fd = gctx->peer_fds[dst_rank];
    if (fd < 0) {
        LOG_ERROR("Gloo send: no connection to rank %d", dst_rank);
        return -1;
    }

    size_t data_size = tensor->numel * sizeof(float);

    if (send_all(fd, &tag, sizeof(tag)) != 0) {
        LOG_ERROR("Gloo send: failed to send tag to rank %d", dst_rank);
        return -1;
    }
    if (send_all(fd, &data_size, sizeof(data_size)) != 0) {
        LOG_ERROR("Gloo send: failed to send size to rank %d", dst_rank);
        return -1;
    }
    if (send_all(fd, tensor->data, data_size) != 0) {
        LOG_ERROR("Gloo send: failed to send data to rank %d", dst_rank);
        return -1;
    }

    LOG_DEBUG("Gloo send: %zu bytes to rank %d (tag=%d)", data_size, dst_rank, tag);
    return 0;
}

/* Protocol: recv tag (int), then data size (size_t), then raw data bytes */
static int gloo_recv(Tensor* tensor, int src_rank, int tag, void* ctx) {
    GlooContext* gctx = get_gloo_ctx(ctx);
    if (!tensor || !tensor->data)
        return -1;

    if (!gctx || gctx->world_size <= 1)
        return 0;

    if (src_rank < 0 || src_rank >= gctx->world_size || src_rank == gctx->rank) {
        LOG_ERROR("Gloo recv: invalid src_rank %d", src_rank);
        return -1;
    }

    int fd = gctx->peer_fds[src_rank];
    if (fd < 0) {
        LOG_ERROR("Gloo recv: no connection to rank %d", src_rank);
        return -1;
    }

    int recv_tag = 0;
    size_t recv_size = 0;

    if (recv_all(fd, &recv_tag, sizeof(recv_tag)) != 0) {
        LOG_ERROR("Gloo recv: failed to receive tag from rank %d", src_rank);
        return -1;
    }
    if (recv_tag != tag) {
        LOG_ERROR("Gloo recv: tag mismatch (expected %d, got %d) from rank %d",
                  tag, recv_tag, src_rank);
        return -1;
    }
    if (recv_all(fd, &recv_size, sizeof(recv_size)) != 0) {
        LOG_ERROR("Gloo recv: failed to receive size from rank %d", src_rank);
        return -1;
    }

    size_t expected_size = tensor->numel * sizeof(float);
    if (recv_size != expected_size) {
        LOG_ERROR("Gloo recv: size mismatch (expected %zu, got %zu) from rank %d",
                  expected_size, recv_size, src_rank);
        return -1;
    }

    if (recv_all(fd, tensor->data, recv_size) != 0) {
        LOG_ERROR("Gloo recv: failed to receive data from rank %d", src_rank);
        return -1;
    }

    LOG_DEBUG("Gloo recv: %zu bytes from rank %d (tag=%d)", recv_size, src_rank, tag);
    return 0;
}

/* Synchronous fallback; wraps result in a completed DistWork handle */
static DistWork* gloo_allreduce_async(Tensor* tensor, DistReduceOp op, void* ctx) {
    DistWork* work = calloc(1, sizeof(DistWork));
    if (!work)
        return NULL;

    work->internal = NULL;
    work->completed = false;
    work->error_code = 0;

    int ret = gloo_allreduce(tensor, op, ctx);
    work->error_code = ret;
    work->completed = true;

    LOG_DEBUG("Gloo allreduce_async completed synchronously (error=%d)", ret);
    return work;
}

static int gloo_wait(DistWork* work) {
    if (!work)
        return -1;

    if (work->completed)
        return work->error_code;

    LOG_ERROR("Gloo wait: work not completed (should not happen)");
    return -1;
}

static int gloo_allreduce(Tensor* tensor, DistReduceOp op, void* ctx) {
    (void)ctx;
    if (!tensor || !tensor->data)
        return -1;

    /* In single-process mode, all-reduce is a no-op */
    DistProcessGroup* group = cml_dist_get_default_group();
    if (!group || group->world_size <= 1) {
        /* Apply averaging if requested */
        if (op == DIST_REDUCE_AVG) {
            float* data = (float*)tensor->data;
            float scale = 1.0f / (float)(group ? group->world_size : 1);
            for (size_t i = 0; i < tensor->numel; i++)
                data[i] *= scale;
        }
        return 0;
    }

    /* For multi-process: use ring all-reduce algorithm */
    int ret = cml_ring_allreduce((float*)tensor->data, tensor->numel,
                                  group->world_size, group->rank,
                                  op, group->ops, group->backend_ctx);
    if (ret != 0) {
        LOG_ERROR("Ring allreduce failed");
        return ret;
    }

    LOG_DEBUG("Gloo allreduce completed (numel: %zu)", tensor->numel);
    return 0;
}

static int gloo_broadcast(Tensor* tensor, int src_rank, void* ctx) {
    (void)ctx;
    (void)src_rank;
    if (!tensor)
        return -1;

    /* In single-process mode, broadcast is a no-op */
    LOG_DEBUG("Gloo broadcast from rank %d (numel: %zu)", src_rank, tensor->numel);
    return 0;
}

static int gloo_allgather(Tensor** output, Tensor* input, void* ctx) {
    (void)ctx;
    if (!output || !input)
        return -1;

    /* Single-process: just copy input to output[0] */
    if (output[0] && output[0]->data && input->data) {
        memcpy(output[0]->data, input->data, input->numel * sizeof(float));
    }

    return 0;
}

static int gloo_reduce_scatter(Tensor* output, Tensor* input, DistReduceOp op, void* ctx) {
    (void)ctx;
    (void)op;
    if (!output || !input)
        return -1;

    /* Single-process: copy relevant chunk */
    if (output->data && input->data) {
        size_t copy_size = output->numel < input->numel ? output->numel : input->numel;
        memcpy(output->data, input->data, copy_size * sizeof(float));
    }

    return 0;
}

static int gloo_barrier(void* ctx) {
    (void)ctx;
    /* Single-process: no-op */
    return 0;
}

static int gloo_init(void* ctx, int world_size, int rank) {
    (void)ctx;

    if (!g_gloo_ctx) {
        LOG_ERROR("Gloo init: context not allocated");
        return -1;
    }

    GlooContext* gctx = g_gloo_ctx;
    gctx->rank = rank;
    gctx->world_size = world_size;

    /* Read environment configuration */
    const char* addr_env = getenv("CML_MASTER_ADDR");
    if (addr_env && addr_env[0] != '\0') {
        strncpy(gctx->master_addr, addr_env, sizeof(gctx->master_addr) - 1);
        gctx->master_addr[sizeof(gctx->master_addr) - 1] = '\0';
    } else {
        strncpy(gctx->master_addr, GLOO_DEFAULT_MASTER_ADDR, sizeof(gctx->master_addr) - 1);
        gctx->master_addr[sizeof(gctx->master_addr) - 1] = '\0';
    }

    const char* port_env = getenv("CML_GLOO_PORT");
    if (port_env && port_env[0] != '\0') {
        gctx->port_base = atoi(port_env);
        if (gctx->port_base <= 0)
            gctx->port_base = GLOO_DEFAULT_PORT_BASE;
    } else {
        gctx->port_base = GLOO_DEFAULT_PORT_BASE;
    }

    /* Allocate peer fd array */
    gctx->peer_fds = calloc((size_t)world_size, sizeof(int));
    if (!gctx->peer_fds)
        return -1;
    for (int i = 0; i < world_size; i++)
        gctx->peer_fds[i] = -1;

    /* Single-process mode: skip socket setup */
    if (world_size <= 1) {
        gctx->listen_fd = -1;
        LOG_INFO("Gloo backend initialized in single-process mode (rank %d/%d)",
                 rank, world_size);
        return 0;
    }

    /* Create listening socket on port_base + rank */
    gctx->listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (gctx->listen_fd < 0) {
        LOG_ERROR("Gloo init: failed to create listen socket: %s", strerror(errno));
        free(gctx->peer_fds);
        gctx->peer_fds = NULL;
        return -1;
    }

    int optval = 1;
    setsockopt(gctx->listen_fd, SOL_SOCKET, SO_REUSEADDR, &optval, sizeof(optval));

    struct sockaddr_in listen_addr;
    memset(&listen_addr, 0, sizeof(listen_addr));
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons((uint16_t)(gctx->port_base + rank));

    if (bind(gctx->listen_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
        LOG_ERROR("Gloo init: failed to bind on port %d: %s",
                  gctx->port_base + rank, strerror(errno));
        close(gctx->listen_fd);
        gctx->listen_fd = -1;
        free(gctx->peer_fds);
        gctx->peer_fds = NULL;
        return -1;
    }

    if (listen(gctx->listen_fd, world_size) < 0) {
        LOG_ERROR("Gloo init: failed to listen: %s", strerror(errno));
        close(gctx->listen_fd);
        gctx->listen_fd = -1;
        free(gctx->peer_fds);
        gctx->peer_fds = NULL;
        return -1;
    }

    LOG_DEBUG("Gloo rank %d listening on port %d", rank, gctx->port_base + rank);

    /*
     * Connection protocol:
     * - Each rank connects to all ranks with a higher index.
     * - Each rank accepts connections from all ranks with a lower index.
     * This avoids deadlocks and ensures a deterministic connection order.
     */

    /* Connect to all higher-ranked peers */
    for (int peer = rank + 1; peer < world_size; peer++) {
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            LOG_ERROR("Gloo init: failed to create socket for peer %d: %s",
                      peer, strerror(errno));
            goto cleanup_error;
        }

        struct sockaddr_in peer_addr;
        memset(&peer_addr, 0, sizeof(peer_addr));
        peer_addr.sin_family = AF_INET;
        peer_addr.sin_port = htons((uint16_t)(gctx->port_base + peer));
        if (inet_pton(AF_INET, gctx->master_addr, &peer_addr.sin_addr) <= 0) {
            LOG_ERROR("Gloo init: invalid master address '%s'", gctx->master_addr);
            close(sock);
            goto cleanup_error;
        }

        /* Retry connection since peer might not be listening yet */
        int connected = 0;
        for (int retry = 0; retry < GLOO_MAX_CONNECT_RETRIES; retry++) {
            if (connect(sock, (struct sockaddr*)&peer_addr, sizeof(peer_addr)) == 0) {
                connected = 1;
                break;
            }
            usleep(GLOO_CONNECT_RETRY_US);
        }

        if (!connected) {
            LOG_ERROR("Gloo init: failed to connect to rank %d at %s:%d: %s",
                      peer, gctx->master_addr, gctx->port_base + peer, strerror(errno));
            close(sock);
            goto cleanup_error;
        }

        /* Send our rank so the peer knows who connected */
        if (send_all(sock, &rank, sizeof(rank)) != 0) {
            LOG_ERROR("Gloo init: failed to send rank to peer %d", peer);
            close(sock);
            goto cleanup_error;
        }

        gctx->peer_fds[peer] = sock;
        LOG_DEBUG("Gloo rank %d connected to rank %d", rank, peer);
    }

    /* Accept connections from all lower-ranked peers */
    for (int i = 0; i < rank; i++) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(gctx->listen_fd, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            LOG_ERROR("Gloo init: failed to accept connection: %s", strerror(errno));
            goto cleanup_error;
        }

        /* Receive the connecting peer's rank */
        int peer_rank = -1;
        if (recv_all(client_fd, &peer_rank, sizeof(peer_rank)) != 0) {
            LOG_ERROR("Gloo init: failed to receive peer rank");
            close(client_fd);
            goto cleanup_error;
        }

        if (peer_rank < 0 || peer_rank >= world_size) {
            LOG_ERROR("Gloo init: received invalid peer rank %d", peer_rank);
            close(client_fd);
            goto cleanup_error;
        }

        gctx->peer_fds[peer_rank] = client_fd;
        LOG_DEBUG("Gloo rank %d accepted connection from rank %d", rank, peer_rank);
    }

    LOG_INFO("Gloo backend initialized (rank %d/%d, port_base=%d, addr=%s)",
             rank, world_size, gctx->port_base, gctx->master_addr);
    return 0;

cleanup_error:
    for (int i = 0; i < world_size; i++) {
        if (gctx->peer_fds[i] >= 0) {
            close(gctx->peer_fds[i]);
            gctx->peer_fds[i] = -1;
        }
    }
    if (gctx->listen_fd >= 0) {
        close(gctx->listen_fd);
        gctx->listen_fd = -1;
    }
    free(gctx->peer_fds);
    gctx->peer_fds = NULL;
    return -1;
}

static void gloo_destroy(void* ctx) {
    (void)ctx;
    GlooContext* gctx = g_gloo_ctx;
    if (!gctx) {
        LOG_INFO("Gloo backend destroyed");
        return;
    }

    if (gctx->peer_fds) {
        for (int i = 0; i < gctx->world_size; i++) {
            if (gctx->peer_fds[i] >= 0)
                close(gctx->peer_fds[i]);
        }
        free(gctx->peer_fds);
        gctx->peer_fds = NULL;
    }

    if (gctx->listen_fd >= 0) {
        close(gctx->listen_fd);
        gctx->listen_fd = -1;
    }

    free(gctx);
    g_gloo_ctx = NULL;
    LOG_INFO("Gloo backend destroyed");
}

DistCommOps* cml_dist_create_gloo_backend(void) {
    DistCommOps* ops = calloc(1, sizeof(DistCommOps));
    if (!ops)
        return NULL;

    GlooContext* gctx = calloc(1, sizeof(GlooContext));
    if (!gctx) {
        free(ops);
        return NULL;
    }
    gctx->listen_fd = -1;
    gctx->peer_fds = NULL;
    g_gloo_ctx = gctx;

    ops->allreduce = gloo_allreduce;
    ops->broadcast = gloo_broadcast;
    ops->allgather = gloo_allgather;
    ops->reduce_scatter = gloo_reduce_scatter;
    ops->barrier = gloo_barrier;
    ops->send = gloo_send;
    ops->recv = gloo_recv;
    ops->allreduce_async = gloo_allreduce_async;
    ops->wait = gloo_wait;
    ops->init = gloo_init;
    ops->destroy = gloo_destroy;
    ops->backend_ctx = gctx;

    return ops;
}
