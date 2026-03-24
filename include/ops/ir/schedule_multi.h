

#ifndef CML_OPS_IR_SCHEDULE_MULTI_H
#define CML_OPS_IR_SCHEDULE_MULTI_H

#include "ops/ir/schedule.h"
#include "backend/device.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    XFER_H2D,  
    XFER_D2H,  
    XFER_D2D,  
} XferDirection;

typedef struct CrossDeviceOp {
    XferDirection direction;
    int           src_device_id;
    int           dst_device_id;
    Tensor*       tensor;          
    size_t        byte_size;
    bool          p2p_possible;    
} CrossDeviceOp;

typedef struct MultiDeviceSchedule {
    CMLSchedule**   device_schedules;
    int*            device_ids;
    int             num_devices;

    CrossDeviceOp*  xfer_ops;
    int             num_xfer_ops;

    struct MultiStep {
        enum { MULTI_STEP_DEVICE, MULTI_STEP_XFER } kind;
        int device_or_xfer_idx;  
    }* steps;
    int num_steps;
} MultiDeviceSchedule;

MultiDeviceSchedule* multi_schedule_build(CMLSchedule* sched,
                                          const int* device_ids,
                                          int num_devices);

void multi_schedule_free(MultiDeviceSchedule* ms);

int multi_schedule_run(MultiDeviceSchedule* ms);

size_t multi_schedule_xfer_bytes(const MultiDeviceSchedule* ms);

int multi_schedule_total_kernels(const MultiDeviceSchedule* ms);

void multi_schedule_print(const MultiDeviceSchedule* ms);

bool devices_p2p_capable(int dev_a, int dev_b);

#ifdef __cplusplus
}
#endif

#endif 
