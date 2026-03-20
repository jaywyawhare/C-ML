#ifndef CML_IB_TRANSPORT_H
#define CML_IB_TRANSPORT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct CMLIBTransport {
    void* ib_ctx;
    void* pd;
    void* cq;
    void** qps;
    int num_peers;
    int rank;
    int world_size;
    void* ib_lib;
    bool connected;
} CMLIBTransport;

bool cml_ib_available(void);
CMLIBTransport* cml_ib_create(int rank, int world_size);
int cml_ib_connect(CMLIBTransport* ib, const char** peer_addrs, int num_peers);
void cml_ib_free(CMLIBTransport* ib);

int cml_ib_send(CMLIBTransport* ib, int peer, const void* buf, size_t size);
int cml_ib_recv(CMLIBTransport* ib, int peer, void* buf, size_t size);
int cml_ib_allreduce(CMLIBTransport* ib, void* buf, size_t size, int elem_size);
int cml_ib_barrier(CMLIBTransport* ib);

typedef struct CMLIBMemReg {
    void* mr;
    void* addr;
    size_t size;
    uint32_t lkey;
    uint32_t rkey;
} CMLIBMemReg;

CMLIBMemReg* cml_ib_register_memory(CMLIBTransport* ib, void* addr, size_t size);
void cml_ib_deregister_memory(CMLIBTransport* ib, CMLIBMemReg* reg);

#ifdef __cplusplus
}
#endif

#endif /* CML_IB_TRANSPORT_H */
