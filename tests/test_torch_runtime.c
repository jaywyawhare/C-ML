#include "torch/torch_c.h"
#include "torch/pte.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif

static int make_temp_pte_path(char* out, size_t out_cap) {
#ifndef _WIN32
    char base[256];
    snprintf(base, sizeof(base), "/tmp/cml_pte_XXXXXX");
    int fd = mkstemp(base);
    if (fd < 0)
        return -1;
    close(fd);
    if (snprintf(out, out_cap, "%s.cpte", base) >= (int)out_cap) {
        unlink(base);
        return -1;
    }
    if (rename(base, out) != 0) {
        unlink(base);
        return -1;
    }
    return 0;
#else
    const char* tmp = getenv("TEMP");
    if (!tmp)
        tmp = getenv("TMP");
    if (!tmp)
        tmp = ".";
    if (snprintf(out, out_cap, "%s\\cml_pte_XXXXXX.cpte", tmp) >= (int)out_cap)
        return -1;
    if (_mktemp(out) == NULL)
        return -1;
    FILE* f = fopen(out, "wb");
    if (!f)
        return -1;
    fclose(f);
    return 0;
#endif
}

static void test_memory_arena(void) {
    printf("  test_memory_arena...");
    TorchMemoryManager* mgr = torch_memory_create(1024 * 1024);
    assert(mgr != NULL);

    void* a = torch_memory_alloc(mgr, 4096);
    void* b = torch_memory_alloc(mgr, 8192);
    assert(a && b);
    assert(torch_memory_used(mgr) >= 4096 + 8192);

    torch_memory_reset(mgr);
    assert(torch_memory_used(mgr) == 0);

    torch_memory_free(mgr);
    printf(" PASSED\n");
}

static void test_memory_arena_exhausted(void) {
    printf("  test_memory_arena_exhausted...");
    TorchMemoryManager* mgr = torch_memory_create(128);
    assert(mgr != NULL);

    void* a = torch_memory_alloc(mgr, 64);
    assert(a != NULL);
    assert(torch_memory_alloc(mgr, 128) == NULL);
    assert(torch_memory_remaining(mgr) < 64);

    torch_memory_free(mgr);
    printf(" PASSED\n");
}

static void test_selective_build(void) {
    printf("  test_selective_build...");
    TorchSelectiveBuildConfig cfg;
    assert(torch_selective_build_from_string("add,mul,matmul,relu", &cfg) == 0);
    assert(cfg.enabled[UOP_ADD]);
    assert(cfg.enabled[UOP_MATMUL]);
    assert(!cfg.enabled[UOP_DIV]);

    torch_selective_build_apply(&cfg);
    assert(torch_selective_build_is_op_enabled(UOP_ADD));
    assert(!torch_selective_build_is_op_enabled(UOP_DIV));

    torch_selective_build_reset();
    assert(torch_selective_build_is_op_enabled(UOP_DIV));
    printf(" PASSED\n");
}

static void test_delegate(void) {
    printf("  test_delegate...");
    TorchDelegate* cpu = torch_delegate_cpu();
    assert(cpu != NULL);
    assert(torch_delegate_find("cpu") != NULL);
    assert(torch_delegate_supports_op(cpu, UOP_ADD));
    assert(!torch_delegate_supports_op(NULL, UOP_ADD));
    printf(" PASSED\n");
}

static void test_pte_roundtrip(void) {
    printf("  test_pte_roundtrip...");

    char path[512];
    assert(make_temp_pte_path(path, sizeof(path)) == 0);

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(4, 2, true));

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {1, 4};
    Tensor* sample = torch_randn(shape, 2, &opts);

    TorchPTEExportOptions export_opts = torch_pte_default_export_options();
    assert(torch_pte_export_module((Module*)model, sample, path, &export_opts) == 0);

    TorchRuntimeModule* rt = torch_runtime_load_pte(path);
    assert(rt != NULL);
    assert(torch_runtime_get_kind(rt) == TORCH_RUNTIME_PTE);
    assert(torch_runtime_has_memory(rt) || torch_runtime_pte_arena_size(rt) == 0);

    Tensor* out = torch_runtime_forward(rt, sample);
    assert(out != NULL);
    assert(torch_tensor_numel(out) == 2);

    torch_tensor_free(out);
    torch_runtime_free(rt);
    torch_tensor_free(sample);
    module_free((Module*)model);
    torch_reset_ir();
    remove(path);

    char sb_path[512];
    snprintf(sb_path, sizeof(sb_path), "%s.kernels", path);
    remove(sb_path);

    printf(" PASSED\n");
}

static void test_pte_export_without_weights(void) {
    printf("  test_pte_export_without_weights...");

    char path[512];
    assert(make_temp_pte_path(path, sizeof(path)) == 0);

    Sequential* model = torch_nn_sequential();
    torch_nn_sequential_add(model, (Module*)torch_nn_linear(4, 2, true));

    TorchTensorOptions opts = torch_options();
    opts = torch_options_dtype(opts, DTYPE_FLOAT32);
    opts = torch_options_device(opts, DEVICE_CPU);

    int shape[] = {1, 4};
    Tensor* sample = torch_randn(shape, 2, &opts);

    TorchPTEExportOptions export_opts = torch_pte_default_export_options();
    export_opts.include_weights = false;
    assert(torch_pte_export_module((Module*)model, sample, path, &export_opts) == 0);

    CMLPTEModel* pte = torch_pte_load(path);
    assert(pte != NULL);
    assert(pte->meta.num_instructions > 0);
    assert(pte->meta.num_constants == 0);
    assert(strcmp(pte->meta.method_name, "forward") == 0);
    torch_pte_free(pte);

    torch_tensor_free(sample);
    module_free((Module*)model);
    torch_reset_ir();
    remove(path);

    char sb_path[512];
    snprintf(sb_path, sizeof(sb_path), "%s.kernels", path);
    remove(sb_path);

    printf(" PASSED\n");
}

int main(void) {
    torch_init();
    printf("Running torch runtime tests:\n");

    test_memory_arena();
    test_memory_arena_exhausted();
    test_selective_build();
    test_delegate();
    test_pte_roundtrip();
    test_pte_export_without_weights();

    torch_cleanup();
    printf("All torch runtime tests passed.\n");
    return 0;
}
