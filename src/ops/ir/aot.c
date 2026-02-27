/**
 * @file aot.c
 * @brief Ahead-of-Time (AOT) compilation implementation
 */

#include "ops/ir/aot.h"
#include "ops/ir/internal.h"
#include "ops/ir/context.h"
#include "core/logging.h"
#include "nn.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <dlfcn.h>
#include <ctype.h>

#ifdef CML_HAS_LLVM_BACKEND
#include "ops/ir/llvm/llvm_backend.h"
#endif

/**
 * @brief Validate a path/argument string to prevent command injection.
 *
 * Rejects strings containing shell metacharacters that could allow
 * arbitrary command execution when passed to popen/system.
 */
static bool aot_validate_path(const char* path) {
    if (!path) return false;
    for (const char* p = path; *p; p++) {
        /* Reject shell metacharacters */
        if (*p == ';' || *p == '|' || *p == '&' || *p == '$' ||
            *p == '`' || *p == '\n' || *p == '\r' ||
            *p == '(' || *p == ')' || *p == '{' || *p == '}' ||
            *p == '<' || *p == '>' || *p == '!' || *p == '\\' ||
            *p == '\'' || *p == '"') {
            LOG_ERROR("Unsafe character '%c' in path: %s", *p, path);
            return false;
        }
    }
    return true;
}

AOTCompileOptions cml_aot_default_options(void) {
    AOTCompileOptions opts = {
        .target_triple = NULL,
        .cpu = NULL,
        .features = NULL,
        .opt_level = AOT_OPT_O3,
        .include_weights = false,
        .generate_header = true,
        .function_name = "cml_model_forward",
        .format = AOT_FORMAT_SHARED_LIB,
        .position_independent = true
    };
    return opts;
}

int cml_aot_compile(CMLGraph_t ir, const char* output_path, const AOTCompileOptions* options) {
#ifdef CML_HAS_LLVM_BACKEND
    if (!ir || !output_path) {
        LOG_ERROR("Invalid arguments to cml_aot_compile");
        return -1;
    }

    /* Validate all user-supplied paths against command injection */
    if (!aot_validate_path(output_path)) {
        LOG_ERROR("Unsafe output path rejected");
        return -1;
    }

    AOTCompileOptions opts = options ? *options : cml_aot_default_options();

    if (opts.target_triple && !aot_validate_path(opts.target_triple)) {
        LOG_ERROR("Unsafe target_triple rejected");
        return -1;
    }
    if (opts.cpu && !aot_validate_path(opts.cpu)) {
        LOG_ERROR("Unsafe cpu option rejected");
        return -1;
    }

    const char* func_name = opts.function_name ? opts.function_name : "cml_model_forward";

    /* AOT compilation via direct LLVM backend — builds LLVM IR from UOps,
       writes to .ll file, then invokes llc for object/shared lib */
    LOG_INFO("AOT compiling IR graph to: %s (via LLVM backend)", output_path);

    /* For now, header-only mode is always supported */
    if (opts.format == AOT_FORMAT_HEADER_ONLY) {
        return cml_aot_generate_header(ir, output_path, func_name);
    }

    /* Full AOT pipeline: build LLVM module from IR, emit to file.
       This requires extracting the IR builder from llvm_backend.c into a
       shared llvm_ir_builder module (planned for Phase 4 follow-up). */
    LOG_ERROR("AOT compilation via LLVM backend not yet fully implemented "
              "(header-only mode works; use cml_aot_compile with AOT_FORMAT_HEADER_ONLY)");
    (void)func_name;
    return -1;

#else
    (void)ir; (void)output_path; (void)options;
    LOG_ERROR("AOT compilation requires LLVM backend support");
    return -1;
#endif
}

int cml_aot_compile_module(struct Module* module, Tensor* sample_input,
                           const char* output_path, const AOTCompileOptions* options) {
    if (!module || !sample_input || !output_path) {
        LOG_ERROR("Invalid arguments to cml_aot_compile_module");
        return -1;
    }

    LOG_INFO("AOT compiling module '%s'", module->name ? module->name : "unnamed");

    /* Trace the forward pass to capture IR */
    cml_ir_reset_global_context();
    Tensor* output = module_forward(module, sample_input);
    if (!output) {
        LOG_ERROR("Forward pass failed during AOT tracing");
        return -1;
    }

    CMLGraph_t ir = cml_ir_get_or_create_context();
    if (!ir) {
        LOG_ERROR("No IR captured during forward pass");
        return -1;
    }

    return cml_aot_compile(ir, output_path, options);
}

CMLAOTModel* cml_aot_load(const char* path) {
    if (!path) {
        LOG_ERROR("NULL path for AOT model load");
        return NULL;
    }

    void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        LOG_ERROR("Failed to load AOT model: %s", dlerror());
        return NULL;
    }

    /* Look up the entry function */
    void* forward_fn = dlsym(handle, "cml_model_forward");
    if (!forward_fn) {
        /* Try alternate name */
        forward_fn = dlsym(handle, "main");
    }

    if (!forward_fn) {
        LOG_ERROR("No entry function found in AOT model");
        dlclose(handle);
        return NULL;
    }

    CMLAOTModel* model = calloc(1, sizeof(CMLAOTModel));
    if (!model) {
        dlclose(handle);
        return NULL;
    }

    model->handle = handle;
    model->forward_fn = forward_fn;
    model->path = strdup(path);

    LOG_INFO("AOT model loaded from: %s", path);
    return model;
}

int cml_aot_execute(CMLAOTModel* model, Tensor** inputs, int num_inputs,
                    Tensor** outputs, int num_outputs) {
    if (!model || !model->forward_fn || !inputs || !outputs) {
        LOG_ERROR("Invalid arguments to cml_aot_execute");
        return -1;
    }

    /* Build memref descriptors for input/output tensors */
    typedef struct {
        float* allocated;
        float* aligned;
        int64_t offset;
        int64_t sizes[1];
        int64_t strides[1];
    } MemRefDesc1D;

    int total = num_inputs + num_outputs;
    MemRefDesc1D* descs = malloc(total * sizeof(MemRefDesc1D));
    if (!descs)
        return -1;

    for (int i = 0; i < total; i++) {
        Tensor* t = i < num_inputs ? inputs[i] : outputs[i - num_inputs];
        if (!t || !t->data) {
            /* Allocate output data if needed */
            if (i >= num_inputs && t && !t->data && t->numel > 0) {
                t->data = calloc(t->numel, sizeof(float));
                t->owns_data = true;
            }
            if (!t || !t->data) {
                free(descs);
                return -1;
            }
        }
        descs[i].allocated = (float*)t->data;
        descs[i].aligned = (float*)t->data;
        descs[i].offset = 0;
        descs[i].sizes[0] = t->numel;
        descs[i].strides[0] = 1;
    }

    /* Call the AOT-compiled function based on argument count */
    typedef void (*Fn3)(MemRefDesc1D*, MemRefDesc1D*, MemRefDesc1D*);
    typedef void (*Fn2)(MemRefDesc1D*, MemRefDesc1D*);

    if (total == 3) {
        Fn3 fn = (Fn3)model->forward_fn;
        fn(&descs[0], &descs[1], &descs[2]);
    } else if (total == 2) {
        Fn2 fn = (Fn2)model->forward_fn;
        fn(&descs[0], &descs[1]);
    } else {
        LOG_WARNING("AOT execute: unsupported arg count %d, using generic call", total);
        /* Generic call via function pointer array */
        void** args = malloc(total * sizeof(void*));
        for (int i = 0; i < total; i++)
            args[i] = &descs[i];
        /* Execute as packed function */
        typedef void (*FnPacked)(void**);
        FnPacked fn = (FnPacked)model->forward_fn;
        fn(args);
        free(args);
    }

    free(descs);
    return 0;
}

void cml_aot_free(CMLAOTModel* model) {
    if (!model)
        return;

    if (model->handle)
        dlclose(model->handle);

    if (model->input_shapes) {
        for (int i = 0; i < model->num_inputs; i++)
            free(model->input_shapes[i]);
        free(model->input_shapes);
    }
    if (model->output_shapes) {
        for (int i = 0; i < model->num_outputs; i++)
            free(model->output_shapes[i]);
        free(model->output_shapes);
    }
    free(model->input_ndims);
    free(model->output_ndims);
    free((char*)model->path);
    free(model);
}

int cml_aot_generate_header(CMLGraph_t ir, const char* header_path, const char* function_name) {
    if (!header_path) {
        LOG_ERROR("NULL header path");
        return -1;
    }

    const char* fname = function_name ? function_name : "cml_model_forward";

    FILE* f = fopen(header_path, "w");
    if (!f) {
        LOG_ERROR("Failed to create header: %s", header_path);
        return -1;
    }

    fprintf(f, "/* Auto-generated CML AOT model header */\n");
    fprintf(f, "#ifndef CML_AOT_MODEL_H\n");
    fprintf(f, "#define CML_AOT_MODEL_H\n\n");
    fprintf(f, "#include <stdint.h>\n\n");
    fprintf(f, "#ifdef __cplusplus\n");
    fprintf(f, "extern \"C\" {\n");
    fprintf(f, "#endif\n\n");

    /* Memref descriptor type */
    fprintf(f, "/* Memref descriptor for 1D float tensor */\n");
    fprintf(f, "typedef struct {\n");
    fprintf(f, "    float* allocated;\n");
    fprintf(f, "    float* aligned;\n");
    fprintf(f, "    int64_t offset;\n");
    fprintf(f, "    int64_t sizes[1];\n");
    fprintf(f, "    int64_t strides[1];\n");
    fprintf(f, "} CMLMemRef1D;\n\n");

    /* Count inputs/outputs from IR */
    int num_inputs = 0, num_outputs = 0;
    if (ir) {
        struct IRNode* node = ir->head;
        while (node) {
            if (node->num_inputs > num_inputs)
                num_inputs = node->num_inputs;
            node = node->next;
        }
        num_outputs = 1; /* Typically one output */
    }

    fprintf(f, "/* Forward declaration of the compiled model function */\n");
    fprintf(f, "/* Inputs: %d, Outputs: %d */\n", num_inputs, num_outputs);
    fprintf(f, "void %s(", fname);
    for (int i = 0; i < num_inputs + num_outputs; i++) {
        if (i > 0) fprintf(f, ", ");
        fprintf(f, "CMLMemRef1D* %s%d",
                i < num_inputs ? "input" : "output",
                i < num_inputs ? i : i - num_inputs);
    }
    fprintf(f, ");\n\n");

    fprintf(f, "#ifdef __cplusplus\n");
    fprintf(f, "}\n");
    fprintf(f, "#endif\n\n");
    fprintf(f, "#endif /* CML_AOT_MODEL_H */\n");

    fclose(f);
    LOG_INFO("Generated AOT header: %s", header_path);
    return 0;
}
