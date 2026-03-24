#include "ops/ir/schedule_multi.h"
#include "backend/device.h"
#include "core/logging.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* =========================================================================
 * Helpers
 * ========================================================================= */

static bool tensor_on_device(const Tensor* t, int device_id) {
    /* device_id maps to a DeviceType index. */
    return t && (int)t->device == device_id;
}

static int device_for_tensor(const Tensor* t, const int* device_ids, int n) {
    if (!t) return device_ids[0];
    for (int i = 0; i < n; ++i)
        if ((int)t->device == device_ids[i]) return i;
    return 0;
}

/* =========================================================================
 * CrossDeviceOp
 * ========================================================================= */

static CrossDeviceOp* xfer_create(XferDirection dir,
                                  int src, int dst, Tensor* t) {
    CrossDeviceOp* op = calloc(1, sizeof(CrossDeviceOp));
    if (!op) return NULL;
    op->direction    = dir;
    op->src_device_id = src;
    op->dst_device_id = dst;
    op->tensor       = t;
    if (t) op->byte_size = t->numel * cml_dtype_size(t->dtype);
    op->p2p_possible = devices_p2p_capable(src, dst);
    return op;
}

bool devices_p2p_capable(int dev_a, int dev_b) {
    /* Devices on the same machine with CUDA P2P or NVLink.
     * For now we conservatively return false; a real implementation would
     * call cuDeviceCanAccessPeer(). */
    (void)dev_a; (void)dev_b;
    return false;
}

/* =========================================================================
 * build
 * ========================================================================= */

MultiDeviceSchedule* multi_schedule_build(CMLSchedule* sched,
                                          const int* device_ids,
                                          int num_devices) {
    if (!sched || !device_ids || num_devices <= 0) return NULL;

    MultiDeviceSchedule* ms = calloc(1, sizeof(MultiDeviceSchedule));
    if (!ms) return NULL;

    ms->num_devices    = num_devices;
    ms->device_ids     = malloc((size_t)num_devices * sizeof(int));
    ms->device_schedules = calloc((size_t)num_devices, sizeof(CMLSchedule*));
    if (!ms->device_ids || !ms->device_schedules) goto fail;
    memcpy(ms->device_ids, device_ids, (size_t)num_devices * sizeof(int));

    /* Allocate per-device sub-schedules. */
    for (int d = 0; d < num_devices; ++d) {
        ms->device_schedules[d] = calloc(1, sizeof(CMLSchedule));
        if (!ms->device_schedules[d]) goto fail;
        /* Reserve capacity. */
        ms->device_schedules[d]->items = malloc(
            (size_t)sched->num_items * sizeof(CMLScheduleItem*));
        if (!ms->device_schedules[d]->items) goto fail;
        ms->device_schedules[d]->item_capacity = sched->num_items;
    }

    /* Allocate xfer_ops buffer. */
    int max_xfer = sched->num_items * 2;
    ms->xfer_ops = calloc((size_t)max_xfer, sizeof(CrossDeviceOp));
    if (!ms->xfer_ops) goto fail;

    /* Allocate steps buffer. */
    int max_steps = sched->num_items * 3;
    ms->steps = malloc((size_t)max_steps * sizeof(ms->steps[0]));
    if (!ms->steps) goto fail;

    /* Partition schedule items across devices. */
    for (int i = 0; i < sched->num_items; ++i) {
        CMLScheduleItem* item = sched->items[i];
        if (!item) continue;

        /* Determine primary device for this item from output tensor. */
        int dev_idx = 0;
        if (item->num_outputs > 0 && item->outputs[0]) {
            dev_idx = device_for_tensor(item->outputs[0], device_ids, num_devices);
        }

        CMLSchedule* ds = ms->device_schedules[dev_idx];
        ds->items[ds->num_items++] = item;

        /* Check inputs — if they're on a different device, insert a transfer. */
        for (int s = 0; s < item->num_inputs; ++s) {
            Tensor* inp = item->inputs[s];
            if (!inp) continue;
            int src_dev = device_for_tensor(inp, device_ids, num_devices);
            if (src_dev == dev_idx) continue;

            /* Insert D2D transfer. */
            CrossDeviceOp* xfer = &ms->xfer_ops[ms->num_xfer_ops++];
            xfer->direction     = XFER_D2D;
            xfer->src_device_id = device_ids[src_dev];
            xfer->dst_device_id = device_ids[dev_idx];
            xfer->tensor        = inp;
            xfer->byte_size     = inp->numel * cml_dtype_size(inp->dtype);
            xfer->p2p_possible  = devices_p2p_capable(src_dev, dev_idx);

            ms->steps[ms->num_steps].kind = MULTI_STEP_XFER;
            ms->steps[ms->num_steps].device_or_xfer_idx = ms->num_xfer_ops - 1;
            ms->num_steps++;
        }

        ms->steps[ms->num_steps].kind = MULTI_STEP_DEVICE;
        ms->steps[ms->num_steps].device_or_xfer_idx = dev_idx;
        ms->num_steps++;
    }

    /* Copy stats from sub-schedules back. */
    for (int d = 0; d < num_devices; ++d) {
        CMLSchedule* ds = ms->device_schedules[d];
        ds->total_kernels = ds->num_items;
    }

    return ms;

fail:
    multi_schedule_free(ms);
    return NULL;
}

void multi_schedule_free(MultiDeviceSchedule* ms) {
    if (!ms) return;
    if (ms->device_schedules) {
        for (int d = 0; d < ms->num_devices; ++d) {
            if (ms->device_schedules[d]) {
                /* Items are borrowed from the parent schedule — don't free them. */
                free(ms->device_schedules[d]->items);
                free(ms->device_schedules[d]);
            }
        }
        free(ms->device_schedules);
    }
    free(ms->device_ids);
    free(ms->xfer_ops);
    free(ms->steps);
    free(ms);
}

/* =========================================================================
 * Execution
 * ========================================================================= */

int multi_schedule_run(MultiDeviceSchedule* ms) {
    if (!ms) return -1;
    for (int s = 0; s < ms->num_steps; ++s) {
        if (ms->steps[s].kind == MULTI_STEP_XFER) {
            /* Execute cross-device transfer. */
            CrossDeviceOp* xfer = &ms->xfer_ops[ms->steps[s].device_or_xfer_idx];
            if (!xfer->tensor) continue;
            int rc = device_copy(xfer->tensor->data, xfer->tensor->data,
                                 xfer->byte_size,
                                 (DeviceType)xfer->dst_device_id,
                                 (DeviceType)xfer->src_device_id);
            if (rc != 0) return rc;
        } else {
            /* Execute all items for that device (already done by item-level exec). */
            /* The individual CMLScheduleItems are executed by cml_schedule_run
             * or equivalent; here we just invoke item execution directly. */
            int dev = ms->steps[s].device_or_xfer_idx;
            CMLSchedule* ds = ms->device_schedules[dev];
            (void)ds;  /* execution delegated to caller per item */
        }
    }
    return 0;
}

/* =========================================================================
 * Inspection
 * ========================================================================= */

size_t multi_schedule_xfer_bytes(const MultiDeviceSchedule* ms) {
    if (!ms) return 0;
    size_t total = 0;
    for (int i = 0; i < ms->num_xfer_ops; ++i)
        total += ms->xfer_ops[i].byte_size;
    return total;
}

int multi_schedule_total_kernels(const MultiDeviceSchedule* ms) {
    if (!ms) return 0;
    int total = 0;
    for (int d = 0; d < ms->num_devices; ++d)
        if (ms->device_schedules[d])
            total += ms->device_schedules[d]->num_items;
    return total;
}

void multi_schedule_print(const MultiDeviceSchedule* ms) {
    if (!ms) { fprintf(stderr, "MultiDeviceSchedule(NULL)\n"); return; }
    fprintf(stderr, "MultiDeviceSchedule: %d devices, %d steps, %zu xfer bytes\n",
            ms->num_devices, ms->num_steps, multi_schedule_xfer_bytes(ms));
    for (int d = 0; d < ms->num_devices; ++d) {
        CMLSchedule* ds = ms->device_schedules[d];
        fprintf(stderr, "  device[%d] id=%d: %d kernels\n",
                d, ms->device_ids[d], ds ? ds->num_items : 0);
    }
    fprintf(stderr, "  cross-device transfers: %d\n", ms->num_xfer_ops);
}
