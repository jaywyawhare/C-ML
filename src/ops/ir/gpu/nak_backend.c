#include "ops/ir/gpu/nak_backend.h"
#include "core/logging.h"

#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h>
#define dlopen(path, flags) ((void*)LoadLibraryA(path))
#define dlsym(handle, sym)  ((void*)GetProcAddress((HMODULE)(handle), (sym)))
#define dlclose(handle)     FreeLibrary((HMODULE)(handle))
#define dlerror()           "dlopen not available"
#define RTLD_LAZY 0
#endif

#ifdef CML_HAS_NAK

typedef void* (*nak_compile_shader_fn)(const void* nir, const void* info);
typedef size_t (*nak_shader_bin_size_fn)(const void* shader);
typedef const void* (*nak_shader_bin_data_fn)(const void* shader);
typedef void (*nak_shader_free_fn)(void* shader);
typedef void* (*nak_from_spirv_fn)(const void* spirv, size_t spirv_size, int gpu_arch);

typedef struct NAKFunctions {
    nak_compile_shader_fn compile_shader;
    nak_shader_bin_size_fn bin_size;
    nak_shader_bin_data_fn bin_data;
    nak_shader_free_fn shader_free;
    nak_from_spirv_fn from_spirv;
} NAKFunctions;

static NAKFunctions g_nak_fns;
static bool g_nak_fns_loaded = false;

static const char* nak_lib_names[] = {
    "libnak.so",
    "libnak.so.0",
    "libnak.so.1",
    NULL
};

static void* try_open_nak(void) {
    for (int i = 0; nak_lib_names[i]; i++) {
        void* lib = dlopen(nak_lib_names[i], RTLD_LAZY);
        if (lib) return lib;
    }

    const char* mesa_path = getenv("CML_NAK_LIB");
    if (mesa_path) {
        void* lib = dlopen(mesa_path, RTLD_LAZY);
        if (lib) return lib;
    }
    return NULL;
}

static int load_nak_symbols(void* lib) {
    if (g_nak_fns_loaded) return 0;

#define LOAD(field, name) do {                          \
    void* _s = dlsym(lib, name);                        \
    if (!_s) return -1;                                 \
    memcpy(&g_nak_fns.field, &_s, sizeof(_s));          \
} while (0)

    LOAD(compile_shader, "nak_compile_shader");
    LOAD(bin_size,       "nak_shader_bin_size");
    LOAD(bin_data,       "nak_shader_bin_data");
    LOAD(shader_free,    "nak_shader_free");
    LOAD(from_spirv,     "nak_from_spirv");

#undef LOAD
    g_nak_fns_loaded = true;
    return 0;
}

bool cml_nak_available(void) {
    void* lib = try_open_nak();
    if (!lib) return false;
    bool ok = (dlsym(lib, "nak_compile_shader") != NULL);
    dlclose(lib);
    return ok;
}

CMLNAKBackend* cml_nak_create(int gpu_arch) {
    CMLNAKBackend* nak = calloc(1, sizeof(CMLNAKBackend));
    if (!nak) return NULL;

    nak->gpu_arch = gpu_arch;
    nak->nak_lib = try_open_nak();

    if (!nak->nak_lib) {
        LOG_INFO("NAK: library not found, backend will be non-functional");
        return nak;
    }

    if (load_nak_symbols(nak->nak_lib) != 0) {
        LOG_WARNING("NAK: missing symbols, backend will be non-functional");
        dlclose(nak->nak_lib);
        nak->nak_lib = NULL;
        return nak;
    }

    nak->initialized = true;
    LOG_INFO("NAK: backend ready for sm_%d", gpu_arch);
    return nak;
}

void cml_nak_free(CMLNAKBackend* nak) {
    if (!nak) return;
    if (nak->nak_lib)
        dlclose(nak->nak_lib);
    free(nak);
}

int cml_nak_compile(CMLNAKBackend* nak, const void* nir_shader,
                    void** binary, size_t* binary_size) {
    if (!nak || !nak->initialized) return -1;
    if (!nir_shader || !binary || !binary_size) return -1;

    *binary = NULL;
    *binary_size = 0;

    void* compiled = g_nak_fns.compile_shader(nir_shader, NULL);
    if (!compiled) {
        LOG_ERROR("NAK: shader compilation failed");
        return -1;
    }

    size_t size = g_nak_fns.bin_size(compiled);
    const void* data = g_nak_fns.bin_data(compiled);

    if (!data || size == 0) {
        LOG_ERROR("NAK: compiled shader has no binary data");
        g_nak_fns.shader_free(compiled);
        return -1;
    }

    void* result = malloc(size);
    if (!result) {
        g_nak_fns.shader_free(compiled);
        return -1;
    }
    memcpy(result, data, size);

    *binary = result;
    *binary_size = size;

    LOG_INFO("NAK: compiled %zu bytes of native GPU code for sm_%d",
             size, nak->gpu_arch);

    g_nak_fns.shader_free(compiled);
    return 0;
}

int cml_nak_compile_spirv(CMLNAKBackend* nak, const void* spirv,
                          size_t spirv_size, void** binary, size_t* binary_size) {
    if (!nak || !nak->initialized) return -1;
    if (!spirv || !spirv_size || !binary || !binary_size) return -1;

    *binary = NULL;
    *binary_size = 0;

    void* nir = g_nak_fns.from_spirv(spirv, spirv_size, nak->gpu_arch);
    if (!nir) {
        LOG_ERROR("NAK: SPIR-V to NIR conversion failed");
        return -1;
    }

    int rc = cml_nak_compile(nak, nir, binary, binary_size);
    return rc;
}

#else /* !CML_HAS_NAK */

bool cml_nak_available(void) { return false; }
CMLNAKBackend* cml_nak_create(int gpu_arch) { (void)gpu_arch; return NULL; }
void cml_nak_free(CMLNAKBackend* nak) { (void)nak; }

int cml_nak_compile(CMLNAKBackend* nak, const void* nir_shader,
                    void** binary, size_t* binary_size) {
    (void)nak; (void)nir_shader; (void)binary; (void)binary_size;
    return -1;
}

int cml_nak_compile_spirv(CMLNAKBackend* nak, const void* spirv,
                          size_t spirv_size, void** binary, size_t* binary_size) {
    (void)nak; (void)spirv; (void)spirv_size; (void)binary; (void)binary_size;
    return -1;
}

#endif
