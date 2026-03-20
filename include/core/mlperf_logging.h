#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void mlperf_log_event(const char* key, const char* value);
void mlperf_log_metric(const char* key, double value);
void mlperf_log_start(const char* benchmark);
void mlperf_log_end(const char* benchmark, const char* status);

#ifdef __cplusplus
}
#endif
