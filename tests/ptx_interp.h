/* A tiny interpreter for the PTX this project's codegen emits.
 *
 * Why this exists: `cml_ptx_gen_*` produces real PTX, and `test_ptx_codegen.c`
 * checks that the *text* contains the instructions it should. Nothing checked the
 * numbers, because that needed an NVIDIA GPU and CI has none -- so an emitted
 * kernel could reference the wrong register, swap two operands, or use the wrong
 * rounding mode and every test would still pass. Interpreting the PTX closes that
 * without hardware: the generated kernel runs here, on the host, and its output
 * is compared against the CPU reference.
 *
 * This is deliberately NOT a general PTX implementation, and it does not belong in
 * the shipped library. It handles exactly the shape this codegen emits:
 *
 *   - one `.visible .entry` per module, `.param .u64` buffers and `.param .u32 n`
 *   - `.reg` banks %p (pred), %r (b32/u32), %rd (b64/u64), %f (f32)
 *   - a straight-line body: the thread-index preamble, a predicated `@%pN ret;`
 *     bounds guard, `ld.global.f32` / `st.global.f32` through computed addresses,
 *     and scalar f32 arithmetic in between
 *
 * Anything outside that (loops, shared memory, real branching, vector types) is
 * reported as unsupported rather than silently skipped -- a quiet no-op here would
 * recreate exactly the false confidence this file is meant to remove.
 *
 * Addresses are synthetic: a u64 holds (buffer_index << 40) | byte_offset, so the
 * pointer arithmetic the kernel performs on a param base decodes back to a real
 * host array without the interpreter needing a flat address space.
 */
#ifndef CML_TEST_PTX_INTERP_H
#define CML_TEST_PTX_INTERP_H

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PTXI_MAX_REG 64
#define PTXI_MAX_BUFS 8
#define PTXI_MAX_PARAMS 8
#define PTXI_ADDR_SHIFT 40

typedef struct {
    const char* name;
    float* data; /* NULL for the scalar n param */
    size_t numel;
} PtxiBuffer;

typedef struct {
    char name[32];
    int buf_index; /* -1 => scalar */
    uint32_t scalar;
} PtxiParam;

typedef struct {
    PtxiBuffer bufs[PTXI_MAX_BUFS];
    int num_bufs;
    PtxiParam params[PTXI_MAX_PARAMS];
    int num_params;

    uint32_t r[PTXI_MAX_REG];
    uint64_t rd[PTXI_MAX_REG];
    float f[PTXI_MAX_REG];
    bool p[PTXI_MAX_REG];

    uint32_t tid, ctaid, ntid;
    char err[256];
} PtxiState;

/* ---- small parsing helpers ------------------------------------------------ */

static char* ptxi_trim(char* s) {
    while (*s && isspace((unsigned char)*s))
        s++;
    char* e = s + strlen(s);
    while (e > s && (isspace((unsigned char)e[-1]) || e[-1] == ';'))
        *--e = '\0';
    return s;
}

/* "%f3" / "%rd0" / "[%rd5]" / "[param_n]" -> bank char and index. */
static bool ptxi_reg(const char* tok, char* bank, int* idx) {
    const char* p = tok;
    while (*p && *p != '%')
        p++;
    if (*p != '%')
        return false;
    p++;
    if (p[0] == 'r' && p[1] == 'd') {
        *bank = 'd';
        p += 2;
    } else if (p[0] == 'r') {
        *bank = 'r';
        p += 1;
    } else if (p[0] == 'f') {
        *bank = 'f';
        p += 1;
    } else if (p[0] == 'p') {
        *bank = 'p';
        p += 1;
    } else {
        return false;
    }
    if (!isdigit((unsigned char)*p))
        return false;
    *idx = atoi(p);
    return *idx >= 0 && *idx < PTXI_MAX_REG;
}

static int ptxi_param_index(PtxiState* st, const char* tok) {
    for (int i = 0; i < st->num_params; i++)
        if (strstr(tok, st->params[i].name))
            return i;
    return -1;
}

static float ptxi_fval(PtxiState* st, const char* tok) {
    char bank;
    int idx;
    if (ptxi_reg(tok, &bank, &idx) && bank == 'f')
        return st->f[idx];
    return strtof(tok, NULL); /* immediate, incl. 0f-hex handled below */
}

/* PTX writes float immediates as 0f3F800000. */
static bool ptxi_hex_float(const char* tok, float* out) {
    const char* p = strstr(tok, "0f");
    if (!p)
        return false;
    uint32_t bits = (uint32_t)strtoul(p + 2, NULL, 16);
    memcpy(out, &bits, sizeof(bits));
    return true;
}

static float ptxi_operand_f(PtxiState* st, const char* tok) {
    float v;
    if (ptxi_hex_float(tok, &v))
        return v;
    return ptxi_fval(st, tok);
}

static uint32_t ptxi_operand_u(PtxiState* st, const char* tok) {
    char bank;
    int idx;
    if (ptxi_reg(tok, &bank, &idx)) {
        if (bank == 'r')
            return st->r[idx];
        if (bank == 'd')
            return (uint32_t)st->rd[idx];
    }
    return (uint32_t)strtoul(tok, NULL, 0);
}

static uint64_t ptxi_operand_u64(PtxiState* st, const char* tok) {
    char bank;
    int idx;
    if (ptxi_reg(tok, &bank, &idx)) {
        if (bank == 'd')
            return st->rd[idx];
        if (bank == 'r')
            return st->r[idx];
    }
    return strtoull(tok, NULL, 0);
}

/* Split "a, b, c" into up to 4 trimmed tokens. */
static int ptxi_split(char* s, char* out[4]) {
    int n     = 0;
    char* tok = strtok(s, ",");
    while (tok && n < 4)
        out[n++] = ptxi_trim(tok), tok = strtok(NULL, ",");
    return n;
}

/* ---- synthetic addressing ------------------------------------------------- */

static bool ptxi_decode_addr(PtxiState* st, uint64_t addr, float** slot) {
    int bi       = (int)(addr >> PTXI_ADDR_SHIFT);
    uint64_t off = addr & (((uint64_t)1 << PTXI_ADDR_SHIFT) - 1);
    if (bi < 0 || bi >= st->num_bufs || !st->bufs[bi].data) {
        snprintf(st->err, sizeof(st->err), "bad buffer id %d in address", bi);
        return false;
    }
    if (off % sizeof(float) != 0) {
        snprintf(st->err, sizeof(st->err), "unaligned address offset %llu",
                 (unsigned long long)off);
        return false;
    }
    size_t i = (size_t)(off / sizeof(float));
    if (i >= st->bufs[bi].numel) {
        snprintf(st->err, sizeof(st->err), "out-of-range access: elem %zu of %zu in '%s'", i,
                 st->bufs[bi].numel, st->bufs[bi].name);
        return false;
    }
    *slot = &st->bufs[bi].data[i];
    return true;
}

/* ---- the interpreter ----------------------------------------------------- */

/* Execute one instruction. Returns 1 to continue, 0 to return from the kernel,
 * -1 on an unsupported/invalid instruction (st->err explains). */
static int ptxi_exec_line(PtxiState* st, char* line) {
    line = ptxi_trim(line);
    if (!*line || line[0] == '/' || line[0] == '.' || line[0] == '{' || line[0] == '}')
        return 1;

    /* Predicated instruction: "@%p0 ret;" */
    if (line[0] == '@') {
        char bank;
        int idx;
        char* sp = strchr(line, ' ');
        if (!sp || !ptxi_reg(line, &bank, &idx) || bank != 'p') {
            snprintf(st->err, sizeof(st->err), "bad predicate: %s", line);
            return -1;
        }
        bool taken = st->p[idx];
        if (line[1] == '!')
            taken = !taken;
        if (!taken)
            return 1;
        return ptxi_exec_line(st, sp + 1);
    }

    if (strncmp(line, "ret", 3) == 0)
        return 0;

    char op[48]  = {0};
    char* sp     = strchr(line, ' ');
    size_t oplen = sp ? (size_t)(sp - line) : strlen(line);
    if (oplen >= sizeof(op)) {
        snprintf(st->err, sizeof(st->err), "opcode too long: %s", line);
        return -1;
    }
    memcpy(op, line, oplen);
    char args_buf[256];
    snprintf(args_buf, sizeof(args_buf), "%s", sp ? sp + 1 : "");
    char* a[4] = {0};
    int na     = ptxi_split(args_buf, a);

    char db;
    int di;
    bool has_dst = na > 0 && ptxi_reg(a[0], &db, &di);

#define NEED(k)                                                                                    \
    if (na < (k)) {                                                                                \
        snprintf(st->err, sizeof(st->err), "expected %d operands: %s", (k), op);                   \
        return -1;                                                                                 \
    }
#define DSTF(v)                                                                                    \
    do {                                                                                           \
        if (!has_dst || db != 'f') {                                                               \
            snprintf(st->err, sizeof(st->err), "expected f32 dst: %s", op);                        \
            return -1;                                                                             \
        }                                                                                          \
        st->f[di] = (v);                                                                           \
    } while (0)

    /* --- moves and the thread-index preamble --- */
    if (strncmp(op, "mov.", 4) == 0) {
        NEED(2);
        if (strstr(a[1], "%tid.x")) {
            st->r[di] = st->tid;
            return 1;
        }
        if (strstr(a[1], "%ctaid.x")) {
            st->r[di] = st->ctaid;
            return 1;
        }
        if (strstr(a[1], "%ntid.x")) {
            st->r[di] = st->ntid;
            return 1;
        }
        if (!has_dst) {
            snprintf(st->err, sizeof(st->err), "mov to non-register");
            return -1;
        }
        if (db == 'f')
            st->f[di] = ptxi_operand_f(st, a[1]);
        else if (db == 'r')
            st->r[di] = ptxi_operand_u(st, a[1]);
        else if (db == 'd')
            st->rd[di] = ptxi_operand_u64(st, a[1]);
        return 1;
    }

    /* --- parameter loads --- */
    if (strncmp(op, "ld.param.", 9) == 0) {
        NEED(2);
        int pi = ptxi_param_index(st, a[1]);
        if (pi < 0) {
            snprintf(st->err, sizeof(st->err), "unknown param %s", a[1]);
            return -1;
        }
        if (st->params[pi].buf_index >= 0) {
            if (!has_dst || db != 'd') {
                snprintf(st->err, sizeof(st->err), "buffer param into non-u64 reg");
                return -1;
            }
            st->rd[di] = (uint64_t)st->params[pi].buf_index << PTXI_ADDR_SHIFT;
        } else {
            if (!has_dst || db != 'r') {
                snprintf(st->err, sizeof(st->err), "scalar param into non-u32 reg");
                return -1;
            }
            st->r[di] = st->params[pi].scalar;
        }
        return 1;
    }

    /* --- global memory --- */
    if (strncmp(op, "ld.global.f32", 13) == 0) {
        NEED(2);
        float* slot = NULL;
        if (!ptxi_decode_addr(st, ptxi_operand_u64(st, a[1]), &slot))
            return -1;
        DSTF(*slot);
        return 1;
    }
    if (strncmp(op, "st.global.f32", 13) == 0) {
        NEED(2);
        float* slot = NULL;
        if (!ptxi_decode_addr(st, ptxi_operand_u64(st, a[0]), &slot))
            return -1;
        *slot = ptxi_operand_f(st, a[1]);
        return 1;
    }

    /* --- integer address arithmetic --- */
    if (strncmp(op, "mad.lo.u32", 10) == 0 || strncmp(op, "mad.lo.s32", 10) == 0) {
        NEED(4);
        st->r[di] = ptxi_operand_u(st, a[1]) * ptxi_operand_u(st, a[2]) + ptxi_operand_u(st, a[3]);
        return 1;
    }
    if (strncmp(op, "cvt.u64.u32", 11) == 0 || strncmp(op, "cvt.s64.s32", 11) == 0) {
        NEED(2);
        st->rd[di] = ptxi_operand_u(st, a[1]);
        return 1;
    }
    if (strncmp(op, "cvt.u32.u64", 11) == 0) {
        NEED(2);
        st->r[di] = (uint32_t)ptxi_operand_u64(st, a[1]);
        return 1;
    }
    if (strncmp(op, "cvt.rmi.f32.f32", 15) == 0) { /* round toward -inf */
        NEED(2);
        DSTF(floorf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "cvt.rpi.f32.f32", 15) == 0) { /* round toward +inf */
        NEED(2);
        DSTF(ceilf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "cvt.rzi.f32.f32", 15) == 0) {
        NEED(2);
        DSTF(truncf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "cvt.rni.f32.f32", 15) == 0) {
        NEED(2);
        DSTF(nearbyintf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "shl.b64", 7) == 0) {
        NEED(3);
        st->rd[di] = ptxi_operand_u64(st, a[1]) << ptxi_operand_u(st, a[2]);
        return 1;
    }
    if (strncmp(op, "shl.b32", 7) == 0) {
        NEED(3);
        st->r[di] = ptxi_operand_u(st, a[1]) << ptxi_operand_u(st, a[2]);
        return 1;
    }
    if (strncmp(op, "add.u64", 7) == 0 || strncmp(op, "add.s64", 7) == 0) {
        NEED(3);
        st->rd[di] = ptxi_operand_u64(st, a[1]) + ptxi_operand_u64(st, a[2]);
        return 1;
    }
    if (strncmp(op, "add.u32", 7) == 0 || strncmp(op, "add.s32", 7) == 0) {
        NEED(3);
        st->r[di] = ptxi_operand_u(st, a[1]) + ptxi_operand_u(st, a[2]);
        return 1;
    }
    if (strncmp(op, "mul.lo.u32", 10) == 0 || strncmp(op, "mul.lo.s32", 10) == 0) {
        NEED(3);
        st->r[di] = ptxi_operand_u(st, a[1]) * ptxi_operand_u(st, a[2]);
        return 1;
    }

    /* --- predicates --- */
    if (strncmp(op, "setp.", 5) == 0) {
        NEED(3);
        if (!has_dst || db != 'p') {
            snprintf(st->err, sizeof(st->err), "setp into non-predicate");
            return -1;
        }
        bool is_f = strstr(op, ".f32") != NULL;
        bool res;
        if (is_f) {
            float x = ptxi_operand_f(st, a[1]), y = ptxi_operand_f(st, a[2]);
            if (strstr(op, "setp.gt"))
                res = x > y;
            else if (strstr(op, "setp.ge"))
                res = x >= y;
            else if (strstr(op, "setp.lt"))
                res = x < y;
            else if (strstr(op, "setp.le"))
                res = x <= y;
            else if (strstr(op, "setp.eq"))
                res = x == y;
            else if (strstr(op, "setp.ne"))
                res = x != y;
            else if (strstr(op, "setp.nan"))
                res = isnan(x) || isnan(y);
            else {
                snprintf(st->err, sizeof(st->err), "unsupported %s", op);
                return -1;
            }
        } else {
            uint32_t x = ptxi_operand_u(st, a[1]), y = ptxi_operand_u(st, a[2]);
            if (strstr(op, "setp.ge"))
                res = x >= y;
            else if (strstr(op, "setp.gt"))
                res = x > y;
            else if (strstr(op, "setp.lt"))
                res = x < y;
            else if (strstr(op, "setp.le"))
                res = x <= y;
            else if (strstr(op, "setp.eq"))
                res = x == y;
            else if (strstr(op, "setp.ne"))
                res = x != y;
            else {
                snprintf(st->err, sizeof(st->err), "unsupported %s", op);
                return -1;
            }
        }
        st->p[di] = res;
        return 1;
    }
    if (strncmp(op, "selp.f32", 8) == 0) {
        NEED(4);
        char pb;
        int pidx;
        if (!ptxi_reg(a[3], &pb, &pidx) || pb != 'p') {
            snprintf(st->err, sizeof(st->err), "selp without predicate");
            return -1;
        }
        DSTF(st->p[pidx] ? ptxi_operand_f(st, a[1]) : ptxi_operand_f(st, a[2]));
        return 1;
    }

    /* --- f32 arithmetic --- */
    if (strncmp(op, "add.f32", 7) == 0) {
        NEED(3);
        DSTF(ptxi_operand_f(st, a[1]) + ptxi_operand_f(st, a[2]));
        return 1;
    }
    if (strncmp(op, "sub.f32", 7) == 0) {
        NEED(3);
        DSTF(ptxi_operand_f(st, a[1]) - ptxi_operand_f(st, a[2]));
        return 1;
    }
    if (strncmp(op, "mul.f32", 7) == 0) {
        NEED(3);
        DSTF(ptxi_operand_f(st, a[1]) * ptxi_operand_f(st, a[2]));
        return 1;
    }
    if (strncmp(op, "div.", 4) == 0 && strstr(op, ".f32")) {
        NEED(3);
        DSTF(ptxi_operand_f(st, a[1]) / ptxi_operand_f(st, a[2]));
        return 1;
    }
    if (strncmp(op, "fma.rn.f32", 10) == 0 || strncmp(op, "mad.f32", 7) == 0) {
        NEED(4);
        DSTF(fmaf(ptxi_operand_f(st, a[1]), ptxi_operand_f(st, a[2]), ptxi_operand_f(st, a[3])));
        return 1;
    }
    if (strncmp(op, "neg.f32", 7) == 0) {
        NEED(2);
        DSTF(-ptxi_operand_f(st, a[1]));
        return 1;
    }
    if (strncmp(op, "abs.f32", 7) == 0) {
        NEED(2);
        DSTF(fabsf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "max.f32", 7) == 0) {
        NEED(3);
        DSTF(fmaxf(ptxi_operand_f(st, a[1]), ptxi_operand_f(st, a[2])));
        return 1;
    }
    if (strncmp(op, "min.f32", 7) == 0) {
        NEED(3);
        DSTF(fminf(ptxi_operand_f(st, a[1]), ptxi_operand_f(st, a[2])));
        return 1;
    }
    if (strncmp(op, "sqrt.", 5) == 0) {
        NEED(2);
        DSTF(sqrtf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "rsqrt.", 6) == 0) {
        NEED(2);
        DSTF(1.0f / sqrtf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "rcp.", 4) == 0) {
        NEED(2);
        DSTF(1.0f / ptxi_operand_f(st, a[1]));
        return 1;
    }
    if (strncmp(op, "ex2.", 4) == 0) {
        NEED(2);
        DSTF(exp2f(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "lg2.", 4) == 0) {
        NEED(2);
        DSTF(log2f(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "sin.", 4) == 0) {
        NEED(2);
        DSTF(sinf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "cos.", 4) == 0) {
        NEED(2);
        DSTF(cosf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "tanh.", 5) == 0) {
        NEED(2);
        DSTF(tanhf(ptxi_operand_f(st, a[1])));
        return 1;
    }
    if (strncmp(op, "rem.", 4) == 0 && strstr(op, "u32")) {
        NEED(3);
        uint32_t d = ptxi_operand_u(st, a[2]);
        st->r[di]  = d ? ptxi_operand_u(st, a[1]) % d : 0;
        return 1;
    }

#undef NEED
#undef DSTF
    snprintf(st->err, sizeof(st->err), "unsupported instruction: %s", line);
    return -1;
}

/* Parse the parameter list of the single `.visible .entry` and bind each name to
 * a caller-supplied buffer (by position) or to the scalar n. */
static bool ptxi_bind_params(PtxiState* st, const char* ptx, uint32_t n) {
    const char* e = strstr(ptx, ".visible .entry");
    if (!e) {
        snprintf(st->err, sizeof(st->err), "no .visible .entry in PTX");
        return false;
    }
    const char* open  = strchr(e, '(');
    const char* close = open ? strchr(open, ')') : NULL;
    if (!open || !close) {
        snprintf(st->err, sizeof(st->err), "malformed entry parameter list");
        return false;
    }

    char list[512];
    size_t len = (size_t)(close - open - 1);
    if (len >= sizeof(list)) {
        snprintf(st->err, sizeof(st->err), "parameter list too long");
        return false;
    }
    memcpy(list, open + 1, len);
    list[len] = '\0';

    int bi    = 0;
    char* tok = strtok(list, ",");
    while (tok && st->num_params < PTXI_MAX_PARAMS) {
        char* t     = ptxi_trim(tok);
        bool is_u64 = strstr(t, ".u64") != NULL;
        /* name is the last whitespace-separated word */
        char* name = t + strlen(t);
        while (name > t && !isspace((unsigned char)name[-1]))
            name--;

        PtxiParam* p = &st->params[st->num_params++];
        snprintf(p->name, sizeof(p->name), "%s", name);
        if (is_u64) {
            if (bi >= st->num_bufs) {
                snprintf(st->err, sizeof(st->err), "kernel wants more buffers than supplied (%d)",
                         st->num_bufs);
                return false;
            }
            p->buf_index = bi++;
            p->scalar    = 0;
        } else {
            p->buf_index = -1;
            p->scalar    = n;
        }
        tok = strtok(NULL, ",");
    }

    if (bi != st->num_bufs) {
        snprintf(st->err, sizeof(st->err), "kernel takes %d buffers, %d supplied", bi,
                 st->num_bufs);
        return false;
    }
    return true;
}

/* Run the kernel over `n` elements with the given block size.
 * Returns true on success; on failure st->err says why. */
static bool ptxi_run(PtxiState* st, const char* ptx, uint32_t n, uint32_t block) {
    st->err[0] = '\0';
    if (!ptxi_bind_params(st, ptx, n))
        return false;

    const char* body = strchr(strstr(ptx, ".visible .entry"), '{');
    if (!body) {
        snprintf(st->err, sizeof(st->err), "no kernel body");
        return false;
    }

    uint32_t grid = (n + block - 1) / block;
    for (uint32_t b = 0; b < grid; b++) {
        for (uint32_t t = 0; t < block; t++) {
            memset(st->r, 0, sizeof(st->r));
            memset(st->rd, 0, sizeof(st->rd));
            memset(st->f, 0, sizeof(st->f));
            memset(st->p, 0, sizeof(st->p));
            st->tid   = t;
            st->ctaid = b;
            st->ntid  = block;

            char buf[256];
            const char* q = body + 1;
            while (*q) {
                const char* nl = strchr(q, '\n');
                size_t l       = nl ? (size_t)(nl - q) : strlen(q);
                if (l >= sizeof(buf)) {
                    snprintf(st->err, sizeof(st->err), "line too long");
                    return false;
                }
                memcpy(buf, q, l);
                buf[l] = '\0';

                int rc = ptxi_exec_line(st, buf);
                if (rc < 0)
                    return false;
                if (rc == 0)
                    break; /* ret */
                if (!nl)
                    break;
                q = nl + 1;
            }
        }
    }
    return true;
}

static void ptxi_add_buffer(PtxiState* st, const char* name, float* data, size_t numel) {
    if (st->num_bufs >= PTXI_MAX_BUFS)
        return;
    st->bufs[st->num_bufs].name  = name;
    st->bufs[st->num_bufs].data  = data;
    st->bufs[st->num_bufs].numel = numel;
    st->num_bufs++;
}

#endif /* CML_TEST_PTX_INTERP_H */
