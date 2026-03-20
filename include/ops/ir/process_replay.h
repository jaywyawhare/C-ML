#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void cml_process_replay_enable(const char* output_dir);
void cml_process_replay_disable(void);
void cml_process_replay_record(const char* kernel_name, const char* source, size_t source_len);
int cml_process_replay_compare(const char* output_dir, const char* baseline_dir);

#ifdef __cplusplus
}
#endif
