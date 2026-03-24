/*
 * schedule_multi — multi-device kernel scheduling.
 *
 * When a computation graph spans multiple devices, this module partitions
 * the CMLSchedule into per-device sub-schedules and inserts explicit
 * cross-device copy operations (H2D, D2D, D2H) at the boundaries.
 *
 * Tensor sharding hints (from tensor/shard.h) are respected when assigning
 * ops to devices.
 */

#ifndef CML_OPS_IR_SCHEDULE_MULTI_H
#define CML_OPS_IR_SCHEDULE_MULTI_H

#include "ops/ir/schedule.h"
#include "backend/device.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------
 * Cross-device transfer descriptor
 * ------------------------------------------------------------------------- */

typedef enum {
    XFER_H2D,  /* host → device */
    XFER_D2H,  /* device → host */
    XFER_D2D,  /* device → device (P2P or via host) */
} XferDirection;

typedef struct CrossDeviceOp {
    XferDirection direction;
    int           src_device_id;
    int           dst_device_id;
    Tensor*       tensor;          /* tensor being transferred */
    size_t        byte_size;
    bool          p2p_possible;    /* true if PCIe P2P or NVLink available */
} CrossDeviceOp;

/* -------------------------------------------------------------------------
 * MultiDeviceSchedule
 * ------------------------------------------------------------------------- */

typedef struct MultiDeviceSchedule {
    /* Per-device sub-schedules (device_schedules[i] runs on device i). */
    CMLSchedule**   device_schedules;
    int*            device_ids;
    int             num_devices;

    /* Cross-device transfers inserted between sub-schedules. */
    CrossDeviceOp*  xfer_ops;
    int             num_xfer_ops;

    /* Execution order: interleaved device + transfer steps. */
    struct MultiStep {
        enum { MULTI_STEP_DEVICE, MULTI_STEP_XFER } kind;
        int device_or_xfer_idx;  /* index into device_schedules or xfer_ops */
    }* steps;
    int num_steps;
} MultiDeviceSchedule;

/* ---- Constructor / destructor ---- */

/*
 * Partition a flat CMLSchedule across the given device IDs.
 * Device assignment follows tensor->device annotations; unassigned ops
 * go to device_ids[0] (primary device).
 *
 * Returns NULL on error.
 */
MultiDeviceSchedule* multi_schedule_build(CMLSchedule* sched,
                                          const int* device_ids,
                                          int num_devices);

void multi_schedule_free(MultiDeviceSchedule* ms);

/* ---- Execution ---- */

/* Execute all sub-schedules and transfers in dependency order.
 * Returns 0 on success. */
int multi_schedule_run(MultiDeviceSchedule* ms);

/* ---- Inspection ---- */

/* Total number of cross-device bytes transferred. */
size_t multi_schedule_xfer_bytes(const MultiDeviceSchedule* ms);

/* Total number of kernels across all devices. */
int multi_schedule_total_kernels(const MultiDeviceSchedule* ms);

/* Print a human-readable breakdown per device. */
void multi_schedule_print(const MultiDeviceSchedule* ms);

/* ---- Utilities ---- */

/* Check if two devices can use P2P transfers (avoids host bounce). */
bool devices_p2p_capable(int dev_a, int dev_b);

#ifdef __cplusplus
}
#endif

#endif /* CML_OPS_IR_SCHEDULE_MULTI_H */
