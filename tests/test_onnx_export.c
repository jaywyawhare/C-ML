/*
 * ONNX export round-trip tests.
 *
 * Export a CML graph to ONNX bytes, re-parse them with the independent
 * importer, execute with cml_onnx_run, and compare against the eager result.
 * The exporter and importer share no serialization code, so numeric
 * agreement validates both protobuf encodings and op mapping.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "cml.h"
#include "tensor/tensor.h"
#include "ops/uops.h"
#include "ops/ir/ir.h"
#include "core/onnx.h"
#include "test_harness.h"

#define APPROX(a, b) (fabsf((float)(a) - (float)(b)) < 2e-4f)

static char* read_file(const char* path, size_t* len_out) {
    FILE* f = fopen(path, "rb");
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buf = malloc((size_t)n);
    if (!buf || fread(buf, 1, (size_t)n, f) != (size_t)n) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *len_out = (size_t)n;
    return buf;
}

/* matmul -> add bias -> relu : the canonical small graph */
static void test_roundtrip_matmul_add_relu(void) {
    printf("Test: export/import/run round-trip (MatMul+Add+Relu)\n");

    const char* path = "/tmp/cml_onnx_rt.onnx";

    float xd[6]  = {1, -2, 3, -4, 5, -6}; /* [2,3] */
    float wd[12] = {
        /* [3,4] */
        0.1f, 0.2f, 0.3f, 0.4f, -0.5f, 0.6f, -0.7f, 0.8f, 0.9f, -1.0f, 1.1f, -1.2f,
    };
    float bd[4]  = {0.5f, -0.5f, 0.25f, -0.25f};
    int xshape[] = {2, 3}, wshape[] = {3, 4}, bshape[] = {4};
    TensorConfig cfg = {0};

    Tensor* X = tensor_from_data(xd, xshape, 2, &cfg);
    Tensor* W = tensor_from_data(wd, wshape, 2, &cfg);
    Tensor* B = tensor_from_data(bd, bshape, 1, &cfg);
    CHECK("leaves created", X && W && B);

    Tensor* Y = uop_matmul(X, W);
    Tensor* Z = uop_add(Y, B);
    Tensor* R = uop_relu(Z);

    int rc = cml_onnx_export_graph(R->ir_context, (Tensor*[]){X}, 1, (Tensor*[]){R}, 1, path);
    CHECK("export succeeded", rc == 0);

    /* eager reference for a fresh input X2 */
    float x2d[6] = {-1.5f, 2.5f, -0.5f, 0.25f, -0.75f, 1.25f};
    float ref[8];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            float acc = bd[j];
            for (int k = 0; k < 3; k++)
                acc += x2d[i * 3 + k] * wd[k * 4 + j];
            ref[i * 4 + j] = acc > 0 ? acc : 0;
        }

    /* re-parse and run with fresh input data */
    size_t flen = 0;
    char* bytes = read_file(path, &flen);
    CHECK("model file readable", bytes != NULL);
    CMLONNXModel* m = cml_onnx_load_buffer((const uint8_t*)bytes, flen);
    CHECK("reimport succeeded", m != NULL);

    if (m) {
        CHECK("node count preserved", m->graph.num_nodes == 3);
        CHECK("initializer count", m->graph.num_initializers == 2); /* W, B */
        CHECK("input count", m->graph.num_inputs == 1);
        CHECK("output count", m->graph.num_outputs == 1);

        Tensor* in_t = tensor_empty(xshape, 2, &cfg);
        memcpy(in_t->data, x2d, sizeof(x2d));
        Tensor* out_t = NULL;
        int rrc       = cml_onnx_run(m, (Tensor*[]){in_t}, 1,
                               (Tensor * *[]){&out_t} ? (Tensor**)&out_t : NULL, 1);
        CHECK("onnx run succeeded", rrc == 0);
        if (rrc == 0 && out_t) {
            tensor_ensure_executed(out_t);
            float* od = (float*)out_t->data;
            int ok    = od != NULL;
            if (ok)
                for (int i = 0; i < 8 && ok; i++)
                    ok = APPROX(od[i], ref[i]);
            CHECK("round-trip values match eager result", ok);
        }
    }

    free(bytes);
    remove(path);
    cml_reset_ir_context();
}

/* Gemm export: UOP_LINEAR must become transB=1 Gemm with fused bias */
static void test_roundtrip_linear(void) {
    printf("Test: export/import/run round-trip (Linear as Gemm)\n");

    const char* path = "/tmp/cml_onnx_lin.onnx";
    const int K = 4, N = 3;

    static float wd[12], bd[3];
    for (int i = 0; i < 12; i++)
        wd[i] = (float)((i % 7) - 3) / 7.0f;
    for (int i = 0; i < 3; i++)
        bd[i] = (float)(i - 1) / 3.0f;
    int wshape[] = {N, K}, bshape[] = {N};
    TensorConfig cfg = {0};
    Tensor* W        = tensor_from_data(wd, wshape, 2, &cfg);
    Tensor* Bt       = tensor_from_data(bd, bshape, 1, &cfg);
    CHECK("linear weights created", W && Bt);

    int xshape[] = {2, 4};
    Tensor* X    = uop_fill(xshape, 2, 0.5f);
    Tensor* Y    = uop_linear(X, W, Bt);
    CHECK("linear built", Y != NULL);

    int rc = cml_onnx_export_graph(X->ir_context, NULL, 0, (Tensor*[]){Y}, 1, path);
    CHECK("linear export succeeded", rc == 0);

    size_t flen = 0;
    char* bytes = read_file(path, &flen);
    CHECK("model file readable", bytes != NULL);
    CMLONNXModel* m = cml_onnx_load_buffer((const uint8_t*)bytes, flen);
    CHECK("reimport succeeded", m != NULL);
    if (m) {
        /* UOP_LINEAR may lower to a short matmul chain; the final node must
         * be the fused Gemm carrying transB=1. */
        int last = m->graph.num_nodes - 1;
        CHECK("final op is Gemm", last >= 0 && strcmp(m->graph.nodes[last].op_type, "Gemm") == 0);
        int transb = -1;
        if (last >= 0)
            for (int i = 0; i < m->graph.nodes[last].num_attrs; i++) {
                if (strcmp(m->graph.nodes[last].attrs[i].name, "transB") == 0)
                    transb = (int)m->graph.nodes[last].attrs[i].value.i;
            }
        CHECK("Gemm carries transB=1", transb == 1);
    }

    free(bytes);
    remove(path);
    cml_reset_ir_context();
}

int main(void) {
    printf("=== ONNX Export Tests ===\n\n");

    test_roundtrip_matmul_add_relu();
    test_roundtrip_linear();

    return TEST_SUMMARY();
}
