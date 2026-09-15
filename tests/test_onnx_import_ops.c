/*
 * ONNX import op coverage tests.
 *
 * Each case hand-assembles a minimal ModelProto (the importer's protobuf
 * reader is independent of this writer, so numeric agreement validates both
 * the wire encoding and each op handler), runs it through cml_onnx_run and
 * compares against a straightforward C reference computation.
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

/* ------------------------------------------------------------------ */
/* Minimal protobuf writer                                             */
/* ------------------------------------------------------------------ */

typedef struct {
    uint8_t d[16384];
    size_t  n;
} Buf;

static void bw_varint(Buf *b, uint64_t v)
{
    while (v >= 0x80) {
        b->d[b->n++] = (uint8_t)(0x80 | (v & 0x7f));
        v >>= 7;
    }
    b->d[b->n++] = (uint8_t)v;
}

static void bw_tag(Buf *b, int field, int wire)
{
    bw_varint(b, ((uint64_t)field << 3) | (uint64_t)wire);
}

static void bw_len(Buf *b, int field, const void *data, size_t len)
{
    bw_tag(b, field, 2);
    bw_varint(b, len);
    memcpy(b->d + b->n, data, len);
    b->n += len;
}

static void bw_uv(Buf *b, int field, uint64_t v)
{
    bw_tag(b, field, 0);
    bw_varint(b, v);
}

static void bw_f32(Buf *b, int field, float v)
{
    bw_tag(b, field, 5);
    memcpy(b->d + b->n, &v, 4);
    b->n += 4;
}

static void bw_str(Buf *b, int field, const char *s)
{
    bw_len(b, field, s, strlen(s));
}

/* packed float payload (no tag) */
static void bw_packed_f32(Buf *b, const float *v, int count)
{
    for (int i = 0; i < count; i++) {
        memcpy(b->d + b->n, &v[i], 4);
        b->n += 4;
    }
}

/* packed varint payload (no tag); two's-complement keeps negatives intact
 * across the reader's uint64 -> int64 cast */
static void bw_packed_i64(Buf *b, const int64_t *v, int count)
{
    for (int i = 0; i < count; i++)
        bw_varint(b, (uint64_t)v[i]);
}

/* TensorProto initializer carrying float_data (field 4) */
static Buf init_f32(const char *name, const int64_t *dims, int ndim,
                    const float *data, int count)
{
    Buf b = { .n = 0 }, tmp = { .n = 0 };
    bw_packed_i64(&tmp, dims, ndim);
    bw_len(&b, 1, tmp.d, tmp.n);
    bw_uv(&b, 2, 1); /* FLOAT */
    tmp.n = 0;
    bw_packed_f32(&tmp, data, count);
    bw_len(&b, 4, tmp.d, tmp.n);
    bw_str(&b, 8, name);
    return b;
}

/* TensorProto initializer carrying int64_data (field 7) -- the importer
 * decodes these element-wise into its float-centric buffers */
static Buf init_i64(const char *name, const int64_t *dims, int ndim,
                    const int64_t *data, int count)
{
    Buf b = { .n = 0 }, tmp = { .n = 0 };
    bw_packed_i64(&tmp, dims, ndim);
    bw_len(&b, 1, tmp.d, tmp.n);
    bw_uv(&b, 2, 7); /* INT64 */
    tmp.n = 0;
    bw_packed_i64(&tmp, data, count);
    bw_len(&b, 7, tmp.d, tmp.n);
    bw_str(&b, 8, name);
    return b;
}

static Buf attr_int(const char *name, int64_t v)
{
    Buf b = { .n = 0 };
    bw_str(&b, 1, name);
    bw_uv(&b, 3, (uint64_t)v);
    return b;
}

static Buf attr_float(const char *name, float v)
{
    Buf b = { .n = 0 };
    bw_str(&b, 1, name);
    bw_f32(&b, 2, v);
    return b;
}

static Buf attr_ints(const char *name, const int64_t *v, int count)
{
    Buf b = { .n = 0 }, tmp = { .n = 0 };
    bw_str(&b, 1, name);
    bw_packed_i64(&tmp, v, count);
    bw_len(&b, 8, tmp.d, tmp.n);
    return b;
}

static Buf attr_str(const char *name, const char *s)
{
    Buf b = { .n = 0 };
    bw_str(&b, 1, name);
    bw_len(&b, 4, s, strlen(s));
    return b;
}

static Buf make_node(const char *op_type, const char **ins, int nin,
                     const char **outs, int nout, Buf *attrs, int nattr)
{
    Buf b = { .n = 0 };
    for (int i = 0; i < nin; i++)  bw_str(&b, 1, ins[i]);
    for (int i = 0; i < nout; i++) bw_str(&b, 2, outs[i]);
    bw_str(&b, 4, op_type);
    for (int i = 0; i < nattr; i++)
        bw_len(&b, 5, attrs[i].d, attrs[i].n);
    return b;
}

/* ValueInfoProto; the importer tolerates a missing type payload */
static Buf make_vi(const char *name)
{
    Buf b = { .n = 0 };
    bw_str(&b, 1, name);
    return b;
}

typedef struct {
    CMLONNXModel *m;
    Tensor *outs[8];
    int rc;
} RunResult;

/* Assemble graph -> model bytes, load, execute with a single input "X". */
static RunResult run_model(Buf *nodes, int nn, Buf *inits, int ni,
                           const int64_t *in_dims, int in_ndim,
                           const float *in_data,
                           const char **out_names, int nout)
{
    RunResult rr = { .m = NULL, .rc = -100 };

    Buf g = { .n = 0 };
    for (int i = 0; i < nn; i++)  bw_len(&g, 1, nodes[i].d, nodes[i].n);
    for (int i = 0; i < ni; i++)  bw_len(&g, 5, inits[i].d, inits[i].n);
    Buf vin = make_vi("X");
    bw_len(&g, 11, vin.d, vin.n);
    for (int i = 0; i < nout; i++) {
        Buf vo = make_vi(out_names[i]);
        bw_len(&g, 12, vo.d, vo.n);
    }
    bw_str(&g, 2, "import_test");

    Buf mm = { .n = 0 };
    bw_uv(&mm, 1, 6); /* ir_version */
    bw_len(&mm, 7, g.d, g.n);

    rr.m = cml_onnx_load_buffer(mm.d, mm.n);
    if (!rr.m) return rr;

    int shape[8];
    for (int i = 0; i < in_ndim; i++) shape[i] = (int)in_dims[i];
    TensorConfig cfg = {0};
    Tensor *in_t = tensor_from_data(in_data, shape, in_ndim, &cfg);

    rr.rc = cml_onnx_run(rr.m, &in_t, 1, rr.outs, nout);
    return rr;
}

#define FREE_MODEL(rr)                                       \
    do {                                                     \
        if ((rr).m) cml_onnx_free((rr).m);                   \
        cml_reset_ir_context();                              \
    } while (0)

static bool out_matches(Tensor *t, const float *ref, int count)
{
    if (!t) return false;
    if (tensor_ensure_executed(t) != 0) return false;
    const float *d = (const float *)tensor_data_ptr(t);
    if (!d) return false;
    bool ok = true;
    for (int i = 0; i < count; i++)
        if (!APPROX(d[i], ref[i])) ok = false;
    if (!ok && getenv("ONNX_TEST_DEBUG")) {
        printf("\n    got :");
        for (int i = 0; i < count; i++) printf(" %g", d[i]);
        printf("\n    want:");
        for (int i = 0; i < count; i++) printf(" %g", ref[i]);
        printf("\n");
    }
    return ok;
}

static bool out_shape_is(Tensor *t, const int *shape, int ndim)
{
    if (!t || t->ndim != ndim) return false;
    for (int i = 0; i < ndim; i++)
        if (t->shape[i] != shape[i]) return false;
    return true;
}

/* ------------------------------------------------------------------ */
/* Tests                                                               */
/* ------------------------------------------------------------------ */

static void test_where(void)
{
    printf("Test: Where\n");
    float cd[4] = {1, 0, 0, 1};
    float ad[4] = {10, 20, 30, 40};
    float bd[4] = {-1, -2, -3, -4};
    int64_t dims2[2] = {2, 2};

    Buf inits[3] = {
        init_f32("C", dims2, 2, cd, 4),
        init_f32("A", dims2, 2, ad, 4),
        init_f32("B", dims2, 2, bd, 4),
    };
    const char *ins[] = {"C", "A", "B"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Where", ins, 3, outs, 1, NULL, 0) };

    RunResult rr = run_model(nodes, 1, inits, 3, dims2, 2, ad, outs, 1);
    CHECK("where: run", rr.rc == 0);
    float ref[4] = {10, -2, -3, 40};
    CHECK("where: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 4));
    FREE_MODEL(rr);
}

static void test_expand(void)
{
    printf("Test: Expand\n");
    float xd[3] = {1, 2, 3};
    int64_t xdims[1] = {3};
    int64_t sdims[1] = {2};
    int64_t shape[2] = {2, 3};

    Buf inits[2] = {
        init_f32("X0", xdims, 1, xd, 3),
        init_i64("S", sdims, 1, shape, 2),
    };
    const char *ins[] = {"X0", "S"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Expand", ins, 2, outs, 1, NULL, 0) };

    /* X is the unused graph input placeholder */
    float dummy[1] = {0};
    int64_t ddims[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 2, ddims, 1, dummy, outs, 1);
    CHECK("expand: run", rr.rc == 0);
    float ref[6] = {1, 2, 3, 1, 2, 3};
    CHECK("expand: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 6));
    FREE_MODEL(rr);
}

static void test_tile(void)
{
    printf("Test: Tile\n");
    float xd[4] = {1, 2, 3, 4};   /* [2,2] */
    int64_t dims2[2] = {2, 2};
    int64_t rdims[1] = {2};
    int64_t reps[2] = {2, 1};

    Buf inits[2] = {
        init_f32("D", dims2, 2, xd, 4),
        init_i64("R", rdims, 1, reps, 2),
    };
    const char *ins[] = {"D", "R"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Tile", ins, 2, outs, 1, NULL, 0) };

    float dummy[1] = {0};
    int64_t ddims[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 2, ddims, 1, dummy, outs, 1);
    CHECK("tile: run", rr.rc == 0);
    int shp[2] = {4, 2};
    float ref[8] = {1, 2, 3, 4, 1, 2, 3, 4}; /* rows tiled by repeats {2,1} */
    CHECK("tile: shape", rr.rc == 0 && out_shape_is(rr.outs[0], shp, 2));
    CHECK("tile: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 8));
    FREE_MODEL(rr);
}

static void test_range(void)
{
    printf("Test: Range\n");
    float sv = 1.5f, lv = 4.5f, dv = 1.0f;
    int64_t scalar[1] = {1};
    Buf inits[3] = {
        init_f32("S", scalar, 1, &sv, 1),
        init_f32("L", scalar, 1, &lv, 1),
        init_f32("D", scalar, 1, &dv, 1),
    };
    const char *ins[] = {"S", "L", "D"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Range", ins, 3, outs, 1, NULL, 0) };

    float dummy[1] = {0};
    int64_t ddims[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 3, ddims, 1, dummy, outs, 1);
    CHECK("range: run", rr.rc == 0);
    float ref[3] = {1.5f, 2.5f, 3.5f};
    CHECK("range: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 3));
    FREE_MODEL(rr);
}

static void test_cumsum(void)
{
    printf("Test: CumSum\n");
    float xd[6] = {1, 2, 3, 4, 5, 6}; /* [2,3] */
    int64_t dims2[2] = {2, 3};
    int64_t scalar[1] = {1};

    Buf inits[1] = { init_i64("A", scalar, 1, (int64_t[]){1}, 1) };
    const char *ins[] = {"X", "A"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("CumSum", ins, 2, outs, 1, NULL, 0) };

    RunResult rr = run_model(nodes, 1, inits, 1, dims2, 2, xd, outs, 1);
    CHECK("cumsum: run", rr.rc == 0);
    float ref[6] = {1, 3, 6, 4, 9, 15};
    CHECK("cumsum: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 6));
    FREE_MODEL(rr);
}

static void test_cumsum_variants(void)
{
    printf("Test: CumSum (exclusive/reverse)\n");
    float xd[6] = {1, 2, 3, 4, 5, 6}; /* [2,3], axis 1 */
    int64_t dims2[2] = {2, 3};
    int64_t axis1[1] = {1};
    const char *ins[] = {"X", "A"};
    const char *outs[] = {"Y"};
    float dummy_axis[1]; (void)dummy_axis;

    /* exclusive: out[i] = sum_{j<i} */
    {
        Buf inits[1] = { init_i64("A", axis1, 1, (int64_t[]){1}, 1) };
        Buf attrs[1] = { attr_int("exclusive", 1) };
        Buf nodes[1] = { make_node("CumSum", ins, 2, outs, 1, attrs, 1) };
        RunResult rr = run_model(nodes, 1, inits, 1, dims2, 2, xd, outs, 1);
        float ref[6] = {0, 1, 3, 0, 4, 9};
        CHECK("cumsum exclusive", rr.rc == 0 && out_matches(rr.outs[0], ref, 6));
        FREE_MODEL(rr);
    }
    /* reverse inclusive: out[i] = sum_{j>=i} */
    {
        Buf inits[1] = { init_i64("A", axis1, 1, (int64_t[]){1}, 1) };
        Buf attrs[1] = { attr_int("reverse", 1) };
        Buf nodes[1] = { make_node("CumSum", ins, 2, outs, 1, attrs, 1) };
        RunResult rr = run_model(nodes, 1, inits, 1, dims2, 2, xd, outs, 1);
        float ref[6] = {6, 5, 3, 15, 11, 6};
        CHECK("cumsum reverse", rr.rc == 0 && out_matches(rr.outs[0], ref, 6));
        FREE_MODEL(rr);
    }
    /* reverse + exclusive: out[i] = sum_{j>i} */
    {
        Buf inits[1] = { init_i64("A", axis1, 1, (int64_t[]){1}, 1) };
        Buf attrs[2] = { attr_int("reverse", 1), attr_int("exclusive", 1) };
        Buf nodes[1] = { make_node("CumSum", ins, 2, outs, 1, attrs, 2) };
        RunResult rr = run_model(nodes, 1, inits, 1, dims2, 2, xd, outs, 1);
        float ref[6] = {5, 3, 0, 11, 6, 0};
        CHECK("cumsum reverse+exclusive",
              rr.rc == 0 && out_matches(rr.outs[0], ref, 6));
        FREE_MODEL(rr);
    }
}

static void test_resize_align_corners(void)
{
    printf("Test: Resize (linear, align_corners)\n");
    float xd[4] = {1, 2, 3, 4};      /* [1,1,2,2] */
    int64_t dims4[4] = {1, 1, 2, 2};
    int64_t zdims[1] = {4};
    int64_t sizes[4] = {1, 1, 3, 3};

    Buf attrs[2] = {
        attr_str("mode", "linear"),
        attr_str("coordinate_transformation_mode", "align_corners"),
    };
    Buf inits[2] = {
        init_f32("X0", dims4, 4, xd, 4),
        init_i64("SZ", zdims, 1, sizes, 4),
    };
    const char *ins[] = {"X0", "", "", "SZ"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Resize", ins, 4, outs, 1, attrs, 2) };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 2, dd1, 1, dummy, outs, 1);
    CHECK("resize_align_corners: run", rr.rc == 0);

    /* align_corners: u = o*(in-1)/(out-1) = o*0.5, endpoints coincide */
    float ref[9];
    for (int oy = 0; oy < 3; oy++) {
        float uy = (float)oy * 0.5f;
        int y0 = (int)uy, y1 = (y0 + 1 < 2) ? y0 + 1 : 1;
        float wy = uy - (float)y0;
        for (int ox = 0; ox < 3; ox++) {
            float ux = (float)ox * 0.5f;
            int x0 = (int)ux, x1 = (x0 + 1 < 2) ? x0 + 1 : 1;
            float wx = ux - (float)x0;
            ref[oy * 3 + ox] =
                xd[y0 * 2 + x0] * (1 - wy) * (1 - wx) +
                xd[y0 * 2 + x1] * (1 - wy) * wx +
                xd[y1 * 2 + x0] * wy * (1 - wx) +
                xd[y1 * 2 + x1] * wy * wx;
        }
    }
    CHECK("resize_align_corners: values",
          rr.rc == 0 && out_matches(rr.outs[0], ref, 9));
    FREE_MODEL(rr);
}

static void test_scatter_nd(void)
{
    printf("Test: ScatterND\n");
    float data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    int64_t ddims[1] = {8};
    int64_t idims[2] = {4, 1};
    int64_t indices[4] = {2, 3, 4, 5};
    float upd[4] = {90, 91, 92, 93};
    int64_t udims[1] = {4};

    Buf inits[3] = {
        init_f32("DATA", ddims, 1, data, 8),
        init_i64("IDX", idims, 2, indices, 4),
        init_f32("UPD", udims, 1, upd, 4),
    };
    const char *ins[] = {"DATA", "IDX", "UPD"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("ScatterND", ins, 3, outs, 1, NULL, 0) };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 3, dd1, 1, dummy, outs, 1);
    CHECK("scatternd: run", rr.rc == 0);
    float ref[8] = {0, 1, 90, 91, 92, 93, 6, 7};
    CHECK("scatternd: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 8));

    /* negative index wraps; scalar update on a [2,3] slice */
    float d2[6] = {0};
    int64_t ddims2[2] = {2, 3};
    int64_t neg_idx[1] = {-1};
    int64_t i1dims[2] = {1, 1};
    float u2[3] = {7, 8, 9};
    int64_t u2dims[2] = {1, 3};
    Buf inits2[3] = {
        init_f32("DATA", ddims2, 2, d2, 6),
        init_i64("IDX", i1dims, 2, neg_idx, 1),
        init_f32("UPD", u2dims, 2, u2, 3),
    };
    FREE_MODEL(rr);
    RunResult rr2 = run_model(nodes, 1, inits2, 3, dd1, 1, dummy, outs, 1);
    float ref2[6] = {0, 0, 0, 7, 8, 9};
    CHECK("scatternd: negative index",
          rr2.rc == 0 && out_matches(rr2.outs[0], ref2, 6));
    FREE_MODEL(rr2);
}

static void test_resize_nearest(void)
{
    printf("Test: Resize (nearest)\n");
    float xd[4] = {1, 2, 3, 4};      /* [1,1,2,2] NCHW */
    int64_t dims4[4] = {1, 1, 2, 2};
    int64_t sdims[1] = {4};
    float scales[4] = {1, 1, 2, 2};

    Buf attrs[1] = { attr_str("mode", "nearest") };
    Buf inits[2] = {
        init_f32("X0", dims4, 4, xd, 4),
        init_f32("SC", sdims, 1, scales, 4),
    };
    const char *ins[] = {"X0", "", "SC", ""};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Resize", ins, 4, outs, 1, attrs, 1) };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 2, dd1, 1, dummy, outs, 1);
    CHECK("resize_nearest: run", rr.rc == 0);
    /* half_pixel + round_prefer_floor maps o -> floor((o+0.5)/2): for a
     * 2x2 -> 4x4 upscale the source rows/cols are {0,0,1,1} */
    float ref[16];
    int smap[4] = {0, 0, 1, 1};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            ref[i * 4 + j] = xd[smap[i] * 2 + smap[j]];
    CHECK("resize_nearest: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 16));
    FREE_MODEL(rr);
}

static void test_resize_linear(void)
{
    printf("Test: Resize (linear)\n");
    float xd[4] = {1, 2, 3, 4};      /* [1,1,2,2] */
    int64_t dims4[4] = {1, 1, 2, 2};
    int64_t zdims[1] = {4};
    int64_t sizes[4] = {1, 1, 3, 3};

    Buf attrs[1] = { attr_str("mode", "linear") };
    Buf inits[2] = {
        init_f32("X0", dims4, 4, xd, 4),
        init_i64("SZ", zdims, 1, sizes, 4),
    };
    const char *ins[] = {"X0", "", "", "SZ"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Resize", ins, 4, outs, 1, attrs, 1) };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 2, dd1, 1, dummy, outs, 1);
    CHECK("resize_linear: run", rr.rc == 0);

    /* independent bilinear half_pixel reference */
    float ref[9];
    for (int oy = 0; oy < 3; oy++) {
        float uy = ((float)oy + 0.5f) * (2.0f / 3.0f) - 0.5f;
        if (uy < 0) uy = 0;
        if (uy > 1) uy = 1;
        int y0 = (int)uy, y1 = (y0 + 1 < 2) ? y0 + 1 : 1;
        float wy = uy - (float)y0;
        for (int ox = 0; ox < 3; ox++) {
            float ux = ((float)ox + 0.5f) * (2.0f / 3.0f) - 0.5f;
            if (ux < 0) ux = 0;
            if (ux > 1) ux = 1;
            int x0 = (int)ux, x1 = (x0 + 1 < 2) ? x0 + 1 : 1;
            float wx = ux - (float)x0;
            ref[oy * 3 + ox] =
                xd[y0 * 2 + x0] * (1 - wy) * (1 - wx) +
                xd[y0 * 2 + x1] * (1 - wy) * wx +
                xd[y1 * 2 + x0] * wy * (1 - wx) +
                xd[y1 * 2 + x1] * wy * wx;
        }
    }
    CHECK("resize_linear: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 9));
    FREE_MODEL(rr);
}

static void test_reduce_ops(void)
{
    printf("Test: ReduceMin/Max/Prod/Sum/Mean\n");
    float xd[6] = {1, 5, 2, 7, 3, 6}; /* [2,3] */
    int64_t dims2[2] = {2, 3};
    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    const char *outs[] = {"Y"};

    /* ReduceMin over axis 1, keepdim=0 */
    {
        int64_t axes[1] = {1};
        Buf attrs[2] = {
            attr_ints("axes", axes, 1),
            attr_int("keepdim", 0),
        };
        Buf inits[1] = { init_f32("X0", dims2, 2, xd, 6) };
        const char *ins[] = {"X0"};
        Buf nodes[1] = { make_node("ReduceMin", ins, 1, outs, 1, attrs, 2) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[2] = {1, 3};
        CHECK("reducemin: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 2));
        FREE_MODEL(rr);
    }

    /* ReduceMax over all axes (empty axes, noop_with_empty_axes=0), keepdim=1 */
    {
        Buf attrs[1] = { attr_int("noop_with_empty_axes", 0) };
        Buf inits[1] = { init_f32("X0", dims2, 2, xd, 6) };
        const char *ins[] = {"X0"};
        Buf nodes[1] = { make_node("ReduceMax", ins, 1, outs, 1, attrs, 1) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[1] = {7};
        CHECK("reducemax: all axes", rr.rc == 0 && out_matches(rr.outs[0], ref, 1));
        FREE_MODEL(rr);
    }

    /* ReduceSum via opset>=13 axes input */
    {
        int64_t axes[1] = {0};
        int64_t adims[1] = {1};
        Buf inits[2] = {
            init_f32("X0", dims2, 2, xd, 6),
            init_i64("AXES", adims, 1, axes, 1),
        };
        const char *ins[] = {"X0", "AXES"};
        Buf nodes[1] = { make_node("ReduceSum", ins, 2, outs, 1, NULL, 0) };
        RunResult rr = run_model(nodes, 1, inits, 2, dd1, 1, dummy, outs, 1);
        float ref[3] = {8, 8, 8};
        CHECK("reducesum: axes input", rr.rc == 0 && out_matches(rr.outs[0], ref, 3));
        FREE_MODEL(rr);
    }

    /* ReduceProd over axis 1 */
    {
        int64_t axes[1] = {1};
        Buf attrs[2] = {
            attr_ints("axes", axes, 1),
            attr_int("keepdim", 0),
        };
        Buf inits[1] = { init_f32("X0", dims2, 2, xd, 6) };
        const char *ins[] = {"X0"};
        Buf nodes[1] = { make_node("ReduceProd", ins, 1, outs, 1, attrs, 2) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[2] = {10, 126};
        CHECK("reduceprod: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 2));
        FREE_MODEL(rr);
    }

    /* ReduceMean over axes {0,1} keepdim=1 */
    {
        int64_t axes[2] = {0, 1};
        Buf attrs[1] = { attr_ints("axes", axes, 2) };
        Buf inits[1] = { init_f32("X0", dims2, 2, xd, 6) };
        const char *ins[] = {"X0"};
        Buf nodes[1] = { make_node("ReduceMean", ins, 1, outs, 1, attrs, 1) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[1] = {24.0f / 6.0f};
        CHECK("reducemean: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 1));
        FREE_MODEL(rr);
    }
}

static void test_argmin_argmax(void)
{
    printf("Test: ArgMin/ArgMax\n");
    float xd[6] = {4, 1, 3, 2, 7, 5}; /* [2,3] */
    int64_t dims2[2] = {2, 3};
    Buf attrs_min[1] = { attr_int("axis", 1) };
    Buf attrs_max[1] = { attr_int("axis", 1) };
    Buf inits[1] = { init_f32("X0", dims2, 2, xd, 6) };
    const char *i1[] = {"X0"};
    const char *outs[2] = {"YMIN", "YMAX"};
    Buf nodes[2] = {
        make_node("ArgMin", i1, 1, &outs[0], 1, attrs_min, 1),
        make_node("ArgMax", i1, 1, &outs[1], 1, attrs_max, 1),
    };

    RunResult r2 = run_model(nodes, 2, inits, 1, dims2, 2, xd, outs, 2);
    CHECK("argmin/max: run", r2.rc == 0);
    float refmin[2] = {1, 0}; /* argmin of {4,1,3} = 1, of {2,7,5} = 0 */
    float refmax[2] = {0, 1}; /* argmax of {4,1,3} = 0, of {2,7,5} = 1 */
    CHECK("argmin: values", r2.rc == 0 && out_matches(r2.outs[0], refmin, 2));
    CHECK("argmax: values", r2.rc == 0 && out_matches(r2.outs[1], refmax, 2));
    FREE_MODEL(r2);
}

static void test_activations(void)
{
    printf("Test: Erf/LeakyRelu/Softplus\n");
    float xd[5] = {-1.0f, -0.5f, 0.0f, 0.5f, 1.0f};
    int64_t dims1[1] = {5};
    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    const char *outs[] = {"Y"};

    {
        Buf inits[1] = { init_f32("X0", dims1, 1, xd, 5) };
        const char *ix[] = {"X0"};
        Buf nodes[1] = { make_node("Erf", ix, 1, outs, 1, NULL, 0) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[5];
        for (int i = 0; i < 5; i++) ref[i] = erff(xd[i]);
        CHECK("erf: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 5));
        FREE_MODEL(rr);
    }

    {
        Buf attrs[1] = { attr_float("alpha", 0.1f) };
        Buf inits[1] = { init_f32("X0", dims1, 1, xd, 5) };
        const char *ix[] = {"X0"};
        Buf nodes[1] = { make_node("LeakyRelu", ix, 1, outs, 1, attrs, 1) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[5];
        for (int i = 0; i < 5; i++)
            ref[i] = xd[i] > 0 ? xd[i] : 0.1f * xd[i];
        CHECK("leakyrelu: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 5));
        FREE_MODEL(rr);
    }

    {
        Buf inits[1] = { init_f32("X0", dims1, 1, xd, 5) };
        const char *ix[] = {"X0"};
        Buf nodes[1] = { make_node("Softplus", ix, 1, outs, 1, NULL, 0) };
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
        float ref[5];
        for (int i = 0; i < 5; i++)
            ref[i] = logf(1.0f + expf(xd[i]));
        CHECK("softplus: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 5));
        FREE_MODEL(rr);
    }

    /* PReLU with per-channel slope on [2,3] */
    {
        float slope[3] = {0.5f, 1.0f, 2.0f};
        int64_t sdims[1] = {3};
        float x2d[6] = {-1, 0.5f, -0.25f, 2, -3, 4};
        int64_t dims2[2] = {2, 3};
        Buf inits2[2] = {
            init_f32("X0", dims2, 2, x2d, 6),
            init_f32("SL", sdims, 1, slope, 3),
        };
        const char *ip[] = {"X0", "SL"};
        Buf nodes[1] = { make_node("PReLU", ip, 2, outs, 1, NULL, 0) };
        RunResult rr = run_model(nodes, 1, inits2, 2, dims2, 2, x2d, outs, 1);
        CHECK("prelu: run", rr.rc == 0);
        float ref[6];
        for (int i = 0; i < 6; i++) {
            float v = x2d[i], s = slope[i % 3];
            ref[i] = (v > 0 ? v : 0.0f) + s * (v < 0 ? v : 0.0f);
        }
        CHECK("prelu: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 6));
        FREE_MODEL(rr);
    }
}

static void test_split(void)
{
    printf("Test: Split\n");

    /* equal split of [2,4] along axis 1 into 2 pieces */
    {
        float xd[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        int64_t dims2[2] = {2, 4};
        Buf attrs[1] = { attr_int("axis", 1) };
        Buf inits[1] = { init_f32("X0", dims2, 2, xd, 8) };
        const char *ix[] = {"X0"};
        const char *outs[] = {"S1", "S2"};
        Buf nodes[1] = { make_node("Split", ix, 1, outs, 2, attrs, 1) };

        float dummy[1] = {0};
        int64_t dd1[1] = {1};
        RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 2);
        CHECK("split: run", rr.rc == 0);
        float ref1[4] = {0, 1, 4, 5};
        float ref2[4] = {2, 3, 6, 7};
        CHECK("split: piece1", rr.rc == 0 && out_matches(rr.outs[0], ref1, 4));
        CHECK("split: piece2", rr.rc == 0 && out_matches(rr.outs[1], ref2, 4));
        FREE_MODEL(rr);
    }

    /* explicit split sizes on [4] axis 0: {3,1} */
    {
        float xd[4] = {10, 11, 12, 13};
        int64_t dims1[1] = {4};
        int64_t sdims[1] = {2};
        int64_t split[2] = {3, 1};
        Buf inits[2] = {
            init_f32("X0", dims1, 1, xd, 4),
            init_i64("SP", sdims, 1, split, 2),
        };
        const char *ix[] = {"X0", "SP"};
        const char *outs[] = {"S1", "S2"};
        Buf nodes[1] = { make_node("Split", ix, 2, outs, 2, NULL, 0) };

        float dummy[1] = {0};
        int64_t dd1[1] = {1};
        RunResult rr = run_model(nodes, 1, inits, 2, dd1, 1, dummy, outs, 2);
        float ref1[3] = {10, 11, 12};
        float ref2[1] = {13};
        CHECK("split sized: piece1", rr.rc == 0 && out_matches(rr.outs[0], ref1, 3));
        CHECK("split sized: piece2", rr.rc == 0 && out_matches(rr.outs[1], ref2, 1));
        FREE_MODEL(rr);
    }
}

static void test_elementwise_extras(void)
{
    printf("Test: Pow/Reciprocal/Floor/Ceil/Round/Sign\n");
    float xd[4] = {1.6f, -2.7f, 3.75f, -0.4f};
    int64_t dims1[1] = {4};
    float dummy[1] = {0};
    int64_t dd1[1] = {1};

    /* one model, six outputs sharing the same input */
    const char *outs[6] = {"FL", "CE", "RO", "SI", "RC", "PO"};

    /* Pow needs a second input; give it an exponent initializer */
    float expv = 2.0f;
    int64_t scalar[1] = {1};
    Buf pow_init = init_f32("EX", scalar, 1, &expv, 1);
    const char *pow_ins[] = {"X0", "EX"};
    const char *ix[] = {"X0"};

    Buf inits_all[2] = { init_f32("X0", dims1, 1, xd, 4), pow_init };
    Buf nodes[6] = {
        make_node("Floor",     ix, 1, &outs[0], 1, NULL, 0),
        make_node("Ceil",      ix, 1, &outs[1], 1, NULL, 0),
        make_node("Round",     ix, 1, &outs[2], 1, NULL, 0),
        make_node("Sign",      ix, 1, &outs[3], 1, NULL, 0),
        make_node("Reciprocal",ix, 1, &outs[4], 1, NULL, 0),
        make_node("Pow",       pow_ins, 2, &outs[5], 1, NULL, 0),
    };

    RunResult rr = run_model(nodes, 6, inits_all, 2, dd1, 1, dummy, outs, 6);
    CHECK("elementwise: run", rr.rc == 0);

    float reffl[4]  = {1, -3, 3, -1};
    float refce[4]  = {2, -2, 4, 0};
    float refro[4]  = {2, -3, 4, 0}; /* inputs chosen away from .5 ties */
    float refsi[4]  = {1, -1, 1, -1};
    float refrc[4]  = {1.f / 1.6f, 1.f / -2.7f, 1.f / 3.75f, 1.f / -0.4f};
    float refpo[4]  = {1.6f * 1.6f, 2.7f * 2.7f, 14.0625f, 0.16f};

    CHECK("floor: values",      rr.rc == 0 && out_matches(rr.outs[0], reffl, 4));
    CHECK("ceil: values",       rr.rc == 0 && out_matches(rr.outs[1], refce, 4));
    CHECK("round: values",      rr.rc == 0 && out_matches(rr.outs[2], refro, 4));
    CHECK("sign: values",       rr.rc == 0 && out_matches(rr.outs[3], refsi, 4));
    CHECK("reciprocal: values", rr.rc == 0 && out_matches(rr.outs[4], refrc, 4));
    CHECK("pow: values",        rr.rc == 0 && out_matches(rr.outs[5], refpo, 4));
    FREE_MODEL(rr);
}

static void test_variadic_elw(void)
{
    printf("Test: Min/Max/Sum/Mean (variadic)\n");
    float ad[3] = {1, 5, 2};
    float bd[3] = {4, 1, 6};
    float cd[3] = {3, 3, 3};
    int64_t dims1[1] = {3};
    float dummy[1] = {0};
    int64_t dd1[1] = {1};

    Buf inits[3] = {
        init_f32("A", dims1, 1, ad, 3),
        init_f32("B", dims1, 1, bd, 3),
        init_f32("C", dims1, 1, cd, 3),
    };
    const char *abc[] = {"A", "B", "C"};
    const char *outs[4] = {"MN", "MX", "SM", "ME"};
    Buf nodes[4] = {
        make_node("Min",  abc, 3, &outs[0], 1, NULL, 0),
        make_node("Max",  abc, 3, &outs[1], 1, NULL, 0),
        make_node("Sum",  abc, 3, &outs[2], 1, NULL, 0),
        make_node("Mean", abc, 3, &outs[3], 1, NULL, 0),
    };

    RunResult rr = run_model(nodes, 4, inits, 3, dd1, 1, dummy, outs, 4);
    CHECK("variadic: run", rr.rc == 0);
    float refmn[3] = {1, 1, 2};
    float refmx[3] = {4, 5, 6};
    float refsm[3] = {8, 9, 11};
    float refme[3] = {8.f / 3.f, 3, 11.f / 3.f};
    CHECK("min: values",  rr.rc == 0 && out_matches(rr.outs[0], refmn, 3));
    CHECK("max: values",  rr.rc == 0 && out_matches(rr.outs[1], refmx, 3));
    CHECK("sum: values",  rr.rc == 0 && out_matches(rr.outs[2], refsm, 3));
    CHECK("mean: values", rr.rc == 0 && out_matches(rr.outs[3], refme, 3));
    FREE_MODEL(rr);
}

static void test_gemm_attrs(void)
{
    printf("Test: Gemm (transA/transB/alpha/beta)\n");

    /* baseline: attribute-free Gemm == matmul with B stored [3,4] */
    {
        float Ad[6] = {1, 2, 3, 4, 5, 6};
        float BdT[12]; /* [3,4]: column j = kernel j */
        float Bsrc[12] = {
            1, 0, 2,
            0, 1, 1,
            1, 1, 0,
            2, 0, 1,
        };
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 3; k++)
                BdT[k * 4 + j] = Bsrc[j * 3 + k];
        int64_t adims[2] = {2, 3}, bdims[2] = {3, 4};
        Buf inits[2] = {
            init_f32("AA", adims, 2, Ad, 6),
            init_f32("BB", bdims, 2, BdT, 12),
        };
        const char *ins[] = {"AA", "BB"};
        const char *outs[] = {"Y"};
        Buf nodes[1] = { make_node("Gemm", ins, 2, outs, 1, NULL, 0) };
        float dummy[1] = {0};
        int64_t dd1[1] = {1};
        RunResult rr = run_model(nodes, 1, inits, 2, dd1, 1, dummy, outs, 1);
        float ref[8];
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 4; j++) {
                float acc = 0;
                for (int k = 0; k < 3; k++)
                    acc += Ad[i * 3 + k] * BdT[k * 4 + j];
                ref[i * 4 + j] = acc;
            }
        CHECK("gemm plain: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 8));
        FREE_MODEL(rr);
    }

    /* full-attribute variant */
    float Ad[6] = {1, 2, 3, 4, 5, 6};    /* [2,3] */
    float Bd[12] = {                     /* [4,3], used transposed */
        1, 0, 2,
        0, 1, 1,
        1, 1, 0,
        2, 0, 1,
    };
    float Cd[4] = {1, -1, 0.5f, -0.5f};  /* [4], broadcast over rows */
    int64_t adims[2] = {2, 3}, bdims[2] = {4, 3}, cdims[1] = {4};

    Buf attrs[4] = {
        attr_float("alpha", 2.0f),
        attr_float("beta", 0.5f),
        attr_int("transA", 0),
        attr_int("transB", 1),
    };
    Buf inits[3] = {
        init_f32("AA", adims, 2, Ad, 6),
        init_f32("BB", bdims, 2, Bd, 12),
        init_f32("CC", cdims, 1, Cd, 4),
    };
    const char *ins[] = {"AA", "BB", "CC"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("Gemm", ins, 3, outs, 1, attrs, 4) };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 3, dd1, 1, dummy, outs, 1);
    CHECK("gemm: run", rr.rc == 0);

    /* A [2,3] times B^T [3,4]: B row k is the k-th output column kernel */
    float ref[8];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 4; j++) {
            float acc = 0;
            for (int k = 0; k < 3; k++)
                acc += Ad[i * 3 + k] * Bd[j * 3 + k];
            ref[i * 4 + j] = 2.0f * acc + 0.5f * Cd[j];
        }
    CHECK("gemm: values", rr.rc == 0 && out_matches(rr.outs[0], ref, 8));
    FREE_MODEL(rr);
}

static void test_slice_steps_axes(void)
{
    printf("Test: Slice (axes/steps/negative)\n");
    float xd[24];                       /* [2,3,4] */
    for (int i = 0; i < 24; i++) xd[i] = (float)i;
    int64_t dims3[3] = {2, 3, 4};
    int64_t sdims[1] = {1};

    /* node 1: last dim, step 2 -> elements {0,2} of dim 2 */
    int64_t st1[1] = {0}, en1[1] = {4}, ax1[1] = {2}, sp1[1] = {2};
    /* node 2: middle dim, negative start/end -> cols {0,1} */
    int64_t st2[1] = {-3}, en2[1] = {-1}, ax2[1] = {1};

    Buf inits[8] = {
        init_f32("X0", dims3, 3, xd, 24),
        init_i64("S1", sdims, 1, st1, 1),
        init_i64("E1", sdims, 1, en1, 1),
        init_i64("A1", sdims, 1, ax1, 1),
        init_i64("P1", sdims, 1, sp1, 1),
        init_i64("S2", sdims, 1, st2, 1),
        init_i64("E2", sdims, 1, en2, 1),
        init_i64("A2", sdims, 1, ax2, 1),
    };
    const char *ins1[] = {"X0", "S1", "E1", "A1", "P1"};
    const char *ins2[] = {"X0", "S2", "E2", "A2"};
    const char *outs[] = {"Y1", "Y2"};
    Buf nodes[2] = {
        make_node("Slice", ins1, 5, &outs[0], 1, NULL, 0),
        make_node("Slice", ins2, 4, &outs[1], 1, NULL, 0),
    };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 2, inits, 8, dd1, 1, dummy, outs, 2);
    CHECK("slice: run", rr.rc == 0);

    float ref1[12]; /* [2,3,4] sliced step 2 on last axis -> [2,3,2] = 12 */
    int n = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 4; k += 2)
                ref1[n++] = xd[i * 12 + j * 4 + k];
    CHECK("slice step2: values", rr.rc == 0 && out_matches(rr.outs[0], ref1, 12));

    float ref2[16];
    n = 0;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 4; k++)
                ref2[n++] = xd[i * 12 + j * 4 + k];
    CHECK("slice negative: values", rr.rc == 0 && out_matches(rr.outs[1], ref2, 16));
    FREE_MODEL(rr);
}

static void test_unsupported_op_rejected(void)
{
    printf("Test: unsupported op (LSTM) rejected cleanly\n");
    float xd[4] = {1, 2, 3, 4};
    int64_t dims2[2] = {2, 2};
    Buf inits[1] = { init_f32("X0", dims2, 2, xd, 4) };
    const char *ix[] = {"X0"};
    const char *outs[] = {"Y"};
    Buf nodes[1] = { make_node("LSTM", ix, 1, outs, 1, NULL, 0) };

    float dummy[1] = {0};
    int64_t dd1[1] = {1};
    RunResult rr = run_model(nodes, 1, inits, 1, dd1, 1, dummy, outs, 1);
    CHECK("lstm: run fails with -2", rr.rc == -2);
    FREE_MODEL(rr);
}

int main(void) {
    printf("=== ONNX Import Op Coverage Tests ===\n\n");

    test_where();
    test_expand();
    test_tile();
    test_range();
    test_cumsum();
    test_cumsum_variants();
    test_scatter_nd();
    test_resize_nearest();
    test_resize_linear();
    test_resize_align_corners();
    test_reduce_ops();
    test_argmin_argmax();
    test_activations();
    test_split();
    test_elementwise_extras();
    test_variadic_elw();
    test_gemm_attrs();
    test_slice_steps_axes();
    test_unsupported_op_rejected();

    return TEST_SUMMARY();
}
