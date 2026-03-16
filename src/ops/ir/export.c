/**
 * @file ir_kernel_export.c
 * @brief Export kernel analysis data for visualization
 */

#include "ops/ir/ir.h"
#include "ops/ir/export.h"
#include "ops/ir/internal.h"
#include "core/logging.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void append_json_string(char** buffer, size_t* offset, size_t* capacity, const char* str) {
    if (!str)
        str = "";

    // Ensure capacity (worst case: every char needs escaping as \uXXXX)
    size_t needed = strlen(str) * 6 + 10;
    while (*offset + needed >= *capacity) {
        *capacity *= 2;
        char* new_buffer = realloc(*buffer, *capacity);
        if (!new_buffer)
            return;
        *buffer = new_buffer;
    }

    (*buffer)[(*offset)++] = '"';
    for (const char* p = str; *p; p++) {
        switch (*p) {
        case '"':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = '"';
            break;
        case '\\':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = '\\';
            break;
        case '\n':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = 'n';
            break;
        case '\r':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = 'r';
            break;
        case '\t':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = 't';
            break;
        case '\b':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = 'b';
            break;
        case '\f':
            (*buffer)[(*offset)++] = '\\';
            (*buffer)[(*offset)++] = 'f';
            break;
        default:
            (*buffer)[(*offset)++] = *p;
            break;
        }
    }
    (*buffer)[(*offset)++] = '"';
}

static void append_format(char** buffer, size_t* offset, size_t* capacity, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);

    // Calculate needed size
    va_list args_copy;
    va_copy(args_copy, args);
    int needed = vsnprintf(NULL, 0, fmt, args_copy);
    va_end(args_copy);

    if (needed < 0) {
        va_end(args);
        return;
    }

    // Ensure capacity
    while (*offset + (size_t)needed + 1 >= *capacity) {
        *capacity *= 2;
        char* new_buffer = realloc(*buffer, *capacity);
        if (!new_buffer) {
            va_end(args);
            return;
        }
        *buffer = new_buffer;
    }

    // Write formatted string
    int written = vsnprintf(*buffer + *offset, *capacity - *offset, fmt, args);
    if (written > 0) {
        *offset += (size_t)written;
    }

    va_end(args);
}

static char* generate_kernel_code_snippet(struct IRNode* node) {
    if (!node)
        return NULL;

    size_t buffer_size = 1024;
    char* code         = malloc(buffer_size);
    if (!code)
        return NULL;

    int offset          = 0;
    const char* op_name = uop_type_to_string(node->type);

    // Generate comment
    offset += snprintf(code + offset, buffer_size - (size_t)offset, "// %s: %s = ", op_name,
                       node->output_name ? node->output_name : "out");

    // Generate operation expression
    switch (node->type) {
    case UOP_ADD:
        if (node->num_inputs >= 2) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "%s + %s\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "inputs[0][i] + inputs[1][i];\n}",
                               node->input_names[0], node->input_names[1]);
        }
        break;
    case UOP_MUL:
        if (node->num_inputs >= 2) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "%s * %s\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "inputs[0][i] * inputs[1][i];\n}",
                               node->input_names[0], node->input_names[1]);
        }
        break;
    case UOP_EXP:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "exp(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "expf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_LOG:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "log(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "logf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_SQRT:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "sqrt(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "sqrtf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_SUB:
        if (node->num_inputs >= 2) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "%s - %s\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "inputs[0][i] - inputs[1][i];\n}",
                               node->input_names[0], node->input_names[1]);
        }
        break;
    case UOP_DIV:
        if (node->num_inputs >= 2) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "%s / %s\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "inputs[0][i] / inputs[1][i];\n}",
                               node->input_names[0], node->input_names[1]);
        }
        break;
    case UOP_NEG:
        if (node->num_inputs >= 1) {
            offset += snprintf(
                code + offset, buffer_size - (size_t)offset,
                "-(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = -inputs[0][i];\n}",
                node->input_names[0]);
        }
        break;
    case UOP_RECIP:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "1.0f / %s\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = 1.0f "
                               "/ inputs[0][i];\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_SUM:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "sum(%s)\nfloat sum = 0.0f;\nfor (int i = 0; i < n; i++) {\n    sum "
                               "+= inputs[0][i];\n}\noutputs[0][0] = sum;",
                               node->input_names[0]);
        }
        break;
    case UOP_MEAN:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "mean(%s)\nfloat sum = 0.0f;\nfor (int i = 0; i < n; i++) {\n    "
                               "sum += inputs[0][i];\n}\noutputs[0][0] = sum / n;",
                               node->input_names[0]);
        }
        break;
    case UOP_MAX:
        if (node->num_inputs >= 2) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "max(%s, %s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "fmaxf(inputs[0][i], inputs[1][i]);\n}",
                               node->input_names[0], node->input_names[1]);
        }
        break;
    case UOP_MAX_REDUCE:
        if (node->num_inputs >= 1) {
            offset +=
                snprintf(code + offset, buffer_size - (size_t)offset,
                         "max_reduce(%s)\nfloat m = inputs[0][0];\nfor (int i = 1; i < n; i++) {\n "
                         "   if (inputs[0][i] > m) m = inputs[0][i];\n}\noutputs[0][0] = m;",
                         node->input_names[0]);
        }
        break;
    case UOP_SIGMOID:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "sigmoid(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "1.0f / (1.0f + expf(-inputs[0][i]));\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_TANH:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "tanh(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "tanhf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_SIN:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "sin(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "sinf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_COS:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "cos(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "cosf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_ABS:
        if (node->num_inputs >= 1) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "abs(%s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "fabsf(inputs[0][i]);\n}",
                               node->input_names[0]);
        }
        break;
    case UOP_CMPLT:
        if (node->num_inputs >= 2) {
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "%s < %s\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                               "inputs[0][i] < inputs[1][i] ? 1.0f : 0.0f;\n}",
                               node->input_names[0], node->input_names[1]);
        }
        break;
    case UOP_WHERE:
        if (node->num_inputs >= 3) {
            offset +=
                snprintf(code + offset, buffer_size - (size_t)offset,
                         "where(%s, %s, %s)\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                         "inputs[0][i] != 0.0f ? inputs[1][i] : inputs[2][i];\n}",
                         node->input_names[0], node->input_names[1], node->input_names[2]);
        }
        break;
    case UOP_MATMUL:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "matmul(%s, %s)\n// BLAS: cblas_sgemm(CblasRowMajor, CblasNoTrans, "
                           "CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);",
                           node->input_names[0] ? node->input_names[0] : "A",
                           node->input_names[1] ? node->input_names[1] : "B");
        break;
    case UOP_CONV2D:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "conv2d(%s, %s)\n// 2D convolution: output = input * kernel",
                           node->input_names[0] ? node->input_names[0] : "input",
                           node->input_names[1] ? node->input_names[1] : "kernel");
        break;
    case UOP_RESHAPE:
    case UOP_PERMUTE:
    case UOP_EXPAND:
    case UOP_SLICE:
    case UOP_STRIDE:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "%s(%s)\n// View operation (no data copy in contiguous case)", op_name,
                           node->input_names[0] ? node->input_names[0] : "tensor");
        break;
    default:
        offset += snprintf(
            code + offset, buffer_size - (size_t)offset,
            "%s(...)\n// Generic: for (int i = 0; i < n; i++) outputs[0][i] = op(inputs[...][i]);",
            op_name);
        break;
    }

    return code;
}

static char* generate_fused_kernel_code(FusedKernel* kernel) {
    if (!kernel || kernel->num_ops == 0)
        return NULL;

    size_t buffer_size = 2048;
    char* code         = malloc(buffer_size);
    if (!code)
        return NULL;

    int offset = 0;

    switch (kernel->fusion_type) {
    case FUSION_FMA:
        if (kernel->num_ops >= 2) {
            struct IRNode* mul = kernel->ops[0];
            struct IRNode* add = kernel->ops[1];
            offset +=
                snprintf(code + offset, buffer_size - (size_t)offset,
                         "// Fused FMA: %s = %s * %s + ...\nfor (int i = 0; i < n; i++) {\n    "
                         "outputs[0][i] = fmaf(inputs[0][i], inputs[1][i], inputs[2][i]);\n}",
                         add->output_name ? add->output_name : "out",
                         mul->input_names[0] ? mul->input_names[0] : "a",
                         mul->input_names[1] ? mul->input_names[1] : "b");
        }
        break;
    case FUSION_CHAIN_ELEMENTWISE:
        offset +=
            snprintf(code + offset, buffer_size - (size_t)offset,
                     "// Fused chained kernel: %d operations\nfor (int i = 0; i < n; i++) {\n",
                     kernel->num_ops);
        for (int i = 0; i < kernel->num_ops && i < 5; i++) {
            struct IRNode* op   = kernel->ops[i];
            const char* op_name = uop_type_to_string(op->type);
            offset += snprintf(code + offset, buffer_size - (size_t)offset,
                               "    float t%d = /* %s */;\n", i, op_name);
        }
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "    outputs[0][i] = t%d;\n}", kernel->num_ops - 1);
        break;
    case FUSION_NEG_ADD:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Fused NEG+ADD -> SUB\nfor (int i = 0; i < n; i++) {\n    "
                           "outputs[0][i] = inputs[1][i] - inputs[0][i];\n}");
        break;
    case FUSION_EXP_LOG:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Fused EXP+LOG -> identity (cancels out)\nfor (int i = 0; i < n; "
                           "i++) {\n    outputs[0][i] = inputs[0][i];\n}");
        break;
    case FUSION_MUL_DIV:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Fused MUL+DIV -> identity (if same operand)\nfor (int i = 0; i < n; "
                           "i++) {\n    outputs[0][i] = inputs[0][i];\n}");
        break;
    case FUSION_SQRT_MUL:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Fused SQRT+MUL\nfor (int i = 0; i < n; i++) {\n    outputs[0][i] = "
                           "sqrtf(inputs[0][i]) * inputs[1][i];\n}");
        break;
    case FUSION_EXP_RECIP:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Fused EXP+RECIP -> exp(-x)\nfor (int i = 0; i < n; i++) {\n    "
                           "outputs[0][i] = expf(-inputs[0][i]);\n}");
        break;
    case FUSION_REDUCE_ELEMENTWISE:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Fused reduction + elementwise: %d operations\n// Performs reduction "
                           "then applies elementwise ops",
                           kernel->num_ops);
        break;
    case FUSION_NONE:
    default:
        offset += snprintf(code + offset, buffer_size - (size_t)offset,
                           "// Unfused kernel: %d operations\nfor (int i = 0; i < n; i++) {\n    "
                           "// Sequential operations\n}",
                           kernel->num_ops);
        break;
    }

    return code;
}

static void analyze_usage(CMLGraph_t ir) {
    if (!ir || !ir->head)
        return;

    int count           = 0;
    struct IRNode* node = ir->head;
    while (node) {
        node->is_used   = false;
        node->use_count = 0;
        count++;
        node = node->next;
    }

    if (count == 0)
        return;

    struct IRNode** nodes = malloc(count * sizeof(struct IRNode*));
    if (!nodes)
        return;

    node = ir->head;
    for (int i = 0; i < count; i++) {
        nodes[i] = node;
        node     = node->next;
    }

    // Also mark any node that has no users? No, that's opposite.
    // We assume the last node is the output (loss).
    nodes[count - 1]->is_used = true;

    for (int i = count - 1; i >= 0; i--) {
        struct IRNode* curr = nodes[i];

        // If node is used (or is a side-effect node, though we don't have those yet)
        if (curr->is_used) {
            // Mark inputs as used
            for (int j = 0; j < curr->num_inputs; j++) {
                if (curr->inputs && curr->inputs[j] && curr->inputs[j]->ir_node) {
                    struct IRNode* producer = curr->inputs[j]->ir_node;
                    producer->is_used       = true;
                    producer->use_count++;
                }
            }
        }
    }

    free(nodes);
}

char* cml_ir_export_kernel_analysis(CMLGraph_t ir, bool optimized) {
    if (!ir)
        return NULL;

    // Run analysis
    analyze_usage(ir);

    size_t capacity = 8192;
    char* buffer    = malloc(capacity);
    if (!buffer)
        return NULL;

    size_t offset = 0;

    // Start JSON object
    append_format(&buffer, &offset, &capacity, "{");

    // Count nodes and analyze
    int total_nodes          = 0;
    int dead_nodes           = 0;
    int fused_kernels        = 0;
    int fusion_opportunities = 0;

    struct IRNode* node = ir->head;
    while (node) {
        total_nodes++;
        if (node->is_used == false && node->use_count == 0) {
            dead_nodes++;
        }
        if (node->fused_kernel && node->fused_kernel->ops[0] == node) {
            fused_kernels++;
        }
        node = node->next;
    }

    // Export metadata
    append_format(&buffer, &offset, &capacity,
                  "\"nodeCount\":%d,\"kernelCount\":%d,\"deadNodes\":%d,\"fusedKernels\":%d,"
                  "\"fusionOpportunities\":%d,",
                  total_nodes, total_nodes - dead_nodes, dead_nodes, fused_kernels,
                  fusion_opportunities);

    // Export kernels array
    append_format(&buffer, &offset, &capacity, "\"kernels\":[");

    node              = ir->head;
    int kernel_idx    = 0;
    bool first_kernel = true;

    while (node) {
        // Skip dead nodes if optimized view
        if (optimized && !node->is_used && node->use_count == 0) {
            node = node->next;
            continue;
        }

        if (!first_kernel) {
            append_format(&buffer, &offset, &capacity, ",");
        }
        first_kernel = false;

        append_format(&buffer, &offset, &capacity, "{");
        append_format(&buffer, &offset, &capacity, "\"id\":%d,", kernel_idx++);

        // Kernel name
        append_format(&buffer, &offset, &capacity, "\"name\":");
        char kernel_name[64];
        if (node->fused_kernel && node->fused_kernel->ops[0] == node) {
            snprintf(kernel_name, sizeof(kernel_name), "fused_kernel_%d", kernel_idx);
        } else {
            snprintf(kernel_name, sizeof(kernel_name), "kernel_%s_%d",
                     uop_type_to_string(node->type), kernel_idx);
        }
        append_json_string(&buffer, &offset, &capacity, kernel_name);
        append_format(&buffer, &offset, &capacity, ",");

        // Type
        append_format(&buffer, &offset, &capacity, "\"type\":");
        append_json_string(&buffer, &offset, &capacity, uop_type_to_string(node->type));
        append_format(&buffer, &offset, &capacity, ",");

        // Code
        append_format(&buffer, &offset, &capacity, "\"code\":");
        char* code = NULL;
        if (node->fused_kernel && node->fused_kernel->ops[0] == node) {
            code = generate_fused_kernel_code(node->fused_kernel);
        } else {
            code = generate_kernel_code_snippet(node);
        }
        append_json_string(&buffer, &offset, &capacity, code ? code : "// Code generation failed");
        if (code)
            free(code);
        append_format(&buffer, &offset, &capacity, ",");

        // Inputs
        append_format(&buffer, &offset, &capacity, "\"inputs\":[");

        if (node->fused_kernel && node->fused_kernel->ops[0] == node) {
            // For fused kernels, collect ALL unique inputs from all ops
            // that are not produced by previous ops in the chain
            FusedKernel* fk = node->fused_kernel;
            char* inputs[64]; // Max 64 inputs for now
            int num_inputs = 0;

            // Track produced outputs to identify internal edges
            char* produced[64];
            int num_produced = 0;

            for (int i = 0; i < fk->num_ops; i++) {
                struct IRNode* op = fk->ops[i];

                // Add inputs if not already produced internally
                for (int j = 0; j < op->num_inputs; j++) {
                    char* input      = op->input_names[j];
                    bool is_internal = false;
                    for (int k = 0; k < num_produced; k++) {
                        if (produced[k] && strcmp(input, produced[k]) == 0) {
                            is_internal = true;
                            break;
                        }
                    }

                    if (!is_internal) {
                        // Check if already in inputs list
                        bool exists = false;
                        for (int k = 0; k < num_inputs; k++) {
                            if (inputs[k] && strcmp(input, inputs[k]) == 0) {
                                exists = true;
                                break;
                            }
                        }
                        if (!exists && num_inputs < 64) {
                            inputs[num_inputs++] = input;
                        }
                    }
                }

                // Record output as produced
                if (op->output_name && num_produced < 64) {
                    produced[num_produced++] = op->output_name;
                }
            }

            for (int i = 0; i < num_inputs; i++) {
                if (i > 0)
                    append_format(&buffer, &offset, &capacity, ",");
                append_json_string(&buffer, &offset, &capacity, inputs[i]);
            }
        } else {
            // Single op
            for (int i = 0; i < node->num_inputs; i++) {
                if (i > 0)
                    append_format(&buffer, &offset, &capacity, ",");
                append_json_string(&buffer, &offset, &capacity, node->input_names[i]);
            }
        }
        append_format(&buffer, &offset, &capacity, "],");

        // Output
        append_format(&buffer, &offset, &capacity, "\"output\":");
        if (node->fused_kernel && node->fused_kernel->ops[0] == node) {
            // For fused kernel, output is the output of the LAST op
            FusedKernel* fk = node->fused_kernel;
            if (fk->num_ops > 0) {
                append_json_string(&buffer, &offset, &capacity,
                                   fk->ops[fk->num_ops - 1]->output_name);
            } else {
                append_json_string(&buffer, &offset, &capacity, node->output_name);
            }
        } else {
            append_json_string(&buffer, &offset, &capacity, node->output_name);
        }
        append_format(&buffer, &offset, &capacity, ",");

        // Flags
        append_format(&buffer, &offset, &capacity, "\"isDead\":%s,\"isFused\":%s",
                      (!node->is_used && node->use_count == 0) ? "true" : "false",
                      (node->fused_kernel) ? "true" : "false");

        // Fused Kernel ID (for grouping)
        if (node->fused_kernel) {
            append_format(&buffer, &offset, &capacity, ",\"fusedKernelId\":\"%p\"",
                          (void*)node->fused_kernel);
        }

        // Export sub-ops for fused kernels
        if (node->fused_kernel && node->fused_kernel->ops[0] == node) {
            append_format(&buffer, &offset, &capacity, ",\"ops\":[");
            FusedKernel* fk = node->fused_kernel;
            for (int i = 0; i < fk->num_ops; i++) {
                struct IRNode* sub = fk->ops[i];
                if (i > 0)
                    append_format(&buffer, &offset, &capacity, ",");
                append_format(&buffer, &offset, &capacity, "{");

                // Type
                append_format(&buffer, &offset, &capacity, "\"type\":");
                append_json_string(&buffer, &offset, &capacity, uop_type_to_string(sub->type));
                append_format(&buffer, &offset, &capacity, ",");

                // Inputs
                append_format(&buffer, &offset, &capacity, "\"inputs\":[");
                for (int j = 0; j < sub->num_inputs; j++) {
                    if (j > 0)
                        append_format(&buffer, &offset, &capacity, ",");
                    append_json_string(&buffer, &offset, &capacity, sub->input_names[j]);
                }
                append_format(&buffer, &offset, &capacity, "],");

                // Output
                append_format(&buffer, &offset, &capacity, "\"output\":");
                append_json_string(&buffer, &offset, &capacity, sub->output_name);

                append_format(&buffer, &offset, &capacity, "}");
            }
            append_format(&buffer, &offset, &capacity, "]");
        }

        append_format(&buffer, &offset, &capacity, "}");

        node = node->next;
    }

    append_format(&buffer, &offset, &capacity, "]");
    append_format(&buffer, &offset, &capacity, "}");

    return buffer;
}

char* cml_ir_export_graph_json(CMLGraph_t ir) {
    if (!ir)
        return NULL;

    char* buffer = malloc(1024 * 1024); // 1MB buffer
    if (!buffer)
        return NULL;

    size_t capacity = 1024 * 1024;
    size_t offset   = 0;

    append_format(&buffer, &offset, &capacity, "{");

    struct IRNode* node = ir->head;
    int node_idx        = 0;
    bool first_node     = true;

    // First pass: Map pointers to indices for edge generation
    // We need to know the index of each input node
    // Since we don't have a hash map, we'll do a linear search or assume topological order?
    // Linear search is O(N^2) but N is small (30-50 nodes).
    // Better: We can just use the fact that inputs must be prior nodes.
    // But we need their indices.
    // Let's create a temporary array of pointers to map back.

    int node_count      = 0;
    struct IRNode* temp = ir->head;
    while (temp) {
        node_count++;
        temp = temp->next;
    }

    struct IRNode** nodes = malloc(node_count * sizeof(struct IRNode*));
    if (!nodes) {
        free(buffer);
        return NULL;
    }

    temp = ir->head;
    for (int i = 0; i < node_count; i++) {
        nodes[i] = temp;
        temp     = temp->next;
    }

    // Generate JSON
    node = ir->head;
    while (node) {
        if (!first_node)
            append_format(&buffer, &offset, &capacity, ",");
        first_node = false;

        append_format(&buffer, &offset, &capacity, "\"%d\":{", node_idx);

        // Label
        append_format(&buffer, &offset, &capacity, "\"label\":");
        append_json_string(&buffer, &offset, &capacity, uop_type_to_string(node->type));

        // Flags
        bool is_dead  = (!node->is_used && node->use_count == 0);
        bool is_fused = (node->fused_kernel != NULL);

        append_format(&buffer, &offset, &capacity, ",\"is_dead\":%s,\"is_fused\":%s",
                      is_dead ? "true" : "false", is_fused ? "true" : "false");

        // Fused ID
        if (node->fused_kernel) {
            append_format(&buffer, &offset, &capacity, ",\"fusedKernelId\":\"%p\"",
                          (void*)node->fused_kernel);
        }

        // Edges
        append_format(&buffer, &offset, &capacity, ",\"src\":[");

        bool first_edge = true;
        for (int i = 0; i < node->num_inputs; i++) {
            // Find input index
            if (node->inputs[i] && node->inputs[i]->ir_node) {
                struct IRNode* input_node = node->inputs[i]->ir_node;
                int input_idx             = -1;

                // Reverse search is likely faster as inputs are usually recent
                for (int j = node_idx - 1; j >= 0; j--) {
                    if (nodes[j] == input_node) {
                        input_idx = j;
                        break;
                    }
                }

                if (input_idx != -1) {
                    if (!first_edge)
                        append_format(&buffer, &offset, &capacity, ",");
                    append_format(&buffer, &offset, &capacity, "[%d,\"%d\"]", i, input_idx);
                    first_edge = false;
                }
            }
        }

        append_format(&buffer, &offset, &capacity, "]}");

        node = node->next;
        node_idx++;
    }

    append_format(&buffer, &offset, &capacity, "}");

    free(nodes);
    return buffer;
}
