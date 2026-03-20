#include "ops/ir/gpu/xmx.h"

#include <string.h>
#include <stdlib.h>

#ifdef __linux__
#include <dlfcn.h>
#endif

bool cml_xmx_available(void) {
#ifdef __linux__
    void* ocl = dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (!ocl)
        return false;

    typedef int (*clGetPlatformIDs_fn)(unsigned, void**, unsigned*);
    typedef int (*clGetDeviceIDs_fn)(void*, unsigned long, unsigned, void**, unsigned*);
    typedef int (*clGetDeviceInfo_fn)(void*, unsigned, size_t, void*, size_t*);

    clGetPlatformIDs_fn pGetPlatformIDs = dlsym(ocl, "clGetPlatformIDs");
    clGetDeviceIDs_fn pGetDeviceIDs = dlsym(ocl, "clGetDeviceIDs");
    clGetDeviceInfo_fn pGetDeviceInfo = dlsym(ocl, "clGetDeviceInfo");

    if (!pGetPlatformIDs || !pGetDeviceIDs || !pGetDeviceInfo) {
        dlclose(ocl);
        return false;
    }

    void* platform = NULL;
    unsigned num_platforms = 0;
    if (pGetPlatformIDs(1, &platform, &num_platforms) != 0 || num_platforms == 0) {
        dlclose(ocl);
        return false;
    }

    void* device = NULL;
    unsigned num_devices = 0;
    /* CL_DEVICE_TYPE_GPU = 4 */
    if (pGetDeviceIDs(platform, 4, 1, &device, &num_devices) != 0 || num_devices == 0) {
        dlclose(ocl);
        return false;
    }

    char extensions[4096] = {0};
    /* CL_DEVICE_EXTENSIONS = 0x1060 */
    pGetDeviceInfo(device, 0x1060, sizeof(extensions) - 1, extensions, NULL);

    bool found = (strstr(extensions, "cl_intel_subgroup_matrix_multiply_accumulate") != NULL);

    dlclose(ocl);
    return found;
#else
    return false;
#endif
}

CMLXMXConfig cml_xmx_get_config(void) {
    CMLXMXConfig cfg;
    cfg.dpas_depth = 8;
    cfg.exec_size = 16;
    cfg.ops_per_chan = 8;
    return cfg;
}
