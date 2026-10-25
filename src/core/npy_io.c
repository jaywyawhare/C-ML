/* NumPy .npy reader/writer: magic + version + ASCII header dict + raw C-order data.
 * Little-endian host assumed; big-endian ('>') descriptors are rejected on load. */
#include "core/npy_io.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "alloc/cml_allocator.h"
#include "core/logging.h"

#define NPY_MAX_DIMS 32

static const unsigned char NPY_MAGIC[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};

/* Map a CML dtype to its little-endian NumPy descr string, or NULL if unsupported. */
static const char* dtype_to_descr(DType dt) {
    switch (dt) {
    case DTYPE_FLOAT32:
        return "<f4";
    case DTYPE_FLOAT64:
        return "<f8";
    case DTYPE_FLOAT16:
        return "<f2";
    case DTYPE_INT8:
        return "|i1";
    case DTYPE_UINT8:
        return "|u1";
    case DTYPE_INT16:
        return "<i2";
    case DTYPE_UINT16:
        return "<u2";
    case DTYPE_INT32:
        return "<i4";
    case DTYPE_UINT32:
        return "<u4";
    case DTYPE_INT64:
        return "<i8";
    case DTYPE_UINT64:
        return "<u8";
    case DTYPE_BOOL:
        return "|b1";
    default:
        return NULL;
    }
}

/* Parse a NumPy descr (e.g. "<f4") into a dtype; returns -1 on big-endian or unknown kind. */
static int descr_to_dtype(const char* d, DType* out) {
    char bo = '|';
    if (*d == '<' || *d == '>' || *d == '=' || *d == '|')
        bo = *d++;
    if (bo == '>') {
        LOG_ERROR("npy: big-endian arrays are not supported");
        return -1;
    }
    char kind = d[0];
    int sz    = atoi(d + 1);
    DType dt;
    if (kind == 'f' && sz == 4)
        dt = DTYPE_FLOAT32;
    else if (kind == 'f' && sz == 8)
        dt = DTYPE_FLOAT64;
    else if (kind == 'f' && sz == 2)
        dt = DTYPE_FLOAT16;
    else if (kind == 'i' && sz == 1)
        dt = DTYPE_INT8;
    else if (kind == 'i' && sz == 2)
        dt = DTYPE_INT16;
    else if (kind == 'i' && sz == 4)
        dt = DTYPE_INT32;
    else if (kind == 'i' && sz == 8)
        dt = DTYPE_INT64;
    else if (kind == 'u' && sz == 1)
        dt = DTYPE_UINT8;
    else if (kind == 'u' && sz == 2)
        dt = DTYPE_UINT16;
    else if (kind == 'u' && sz == 4)
        dt = DTYPE_UINT32;
    else if (kind == 'u' && sz == 8)
        dt = DTYPE_UINT64;
    else if (kind == 'b' && sz == 1)
        dt = DTYPE_BOOL;
    else {
        LOG_ERROR("npy: unsupported descr '%s'", d);
        return -1;
    }
    *out = dt;
    return 0;
}

int cml_tensor_save_npy(Tensor* t, const char* path) {
    if (!t || !path)
        return -1;
    const char* descr = dtype_to_descr(t->dtype);
    if (!descr) {
        LOG_ERROR("npy: unsupported dtype for save");
        return -1;
    }
    void* data = tensor_data_ptr(t);
    if (!data)
        return -1;

    char dict[512];
    int off =
        snprintf(dict, sizeof dict, "{'descr': '%s', 'fortran_order': False, 'shape': (", descr);
    for (int i = 0; i < t->ndim && off < (int)sizeof dict; i++)
        off += snprintf(dict + off, sizeof dict - off, "%d%s", t->shape[i],
                        (t->ndim == 1 || i + 1 < t->ndim) ? ", " : "");
    off += snprintf(dict + off, sizeof dict - off, "), }");
    if (off < 0 || off >= (int)sizeof dict)
        return -1;

    size_t pre    = 10;                    /* magic(6) + version(2) + header-length(2) */
    size_t base   = pre + (size_t)off + 1; /* trailing newline */
    size_t padded = (base + 63) & ~(size_t)63;
    size_t npad   = padded - base;
    uint16_t hlen = (uint16_t)(padded - pre);

    FILE* f = fopen(path, "wb");
    if (!f) {
        LOG_ERROR("npy: cannot open '%s' for writing", path);
        return -1;
    }
    unsigned char ver[2] = {1, 0};
    int ok               = fwrite(NPY_MAGIC, 1, 6, f) == 6 && fwrite(ver, 1, 2, f) == 2 &&
             fwrite(&hlen, sizeof hlen, 1, f) == 1 &&
             fwrite(dict, 1, (size_t)off, f) == (size_t)off;
    for (size_t i = 0; ok && i < npad; i++)
        ok = fputc(' ', f) != EOF;
    ok            = ok && fputc('\n', f) != EOF;
    size_t nbytes = t->numel * cml_dtype_size(t->dtype);
    ok            = ok && fwrite(data, 1, nbytes, f) == nbytes;
    fclose(f);
    if (!ok)
        LOG_ERROR("npy: write failed for '%s'", path);
    return ok ? 0 : -1;
}

Tensor* cml_tensor_load_npy(const char* path) {
    if (!path)
        return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) {
        LOG_ERROR("npy: cannot open '%s'", path);
        return NULL;
    }
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f) != 8 || memcmp(hdr, NPY_MAGIC, 6) != 0) {
        LOG_ERROR("npy: bad magic in '%s'", path);
        fclose(f);
        return NULL;
    }
    size_t hlen = 0;
    if (hdr[6] == 1) {
        uint16_t v;
        if (fread(&v, sizeof v, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        hlen = v;
    } else {
        uint32_t v;
        if (fread(&v, sizeof v, 1, f) != 1) {
            fclose(f);
            return NULL;
        }
        hlen = v;
    }
    char* dict = cml_malloc(hlen + 1);
    if (!dict || fread(dict, 1, hlen, f) != hlen) {
        cml_free(dict);
        fclose(f);
        return NULL;
    }
    dict[hlen] = '\0';

    if (strstr(dict, "'fortran_order': True")) {
        LOG_ERROR("npy: Fortran-ordered arrays are not supported");
        cml_free(dict);
        fclose(f);
        return NULL;
    }
    char* dp = strstr(dict, "'descr'");
    char* sp = strstr(dict, "'shape'");
    if (!dp || !sp) {
        cml_free(dict);
        fclose(f);
        return NULL;
    }
    dp       = strchr(dp + 7, '\'');
    char* de = dp ? strchr(dp + 1, '\'') : NULL;
    if (!dp || !de) {
        cml_free(dict);
        fclose(f);
        return NULL;
    }
    *de = '\0';
    DType dtype;
    if (descr_to_dtype(dp + 1, &dtype) != 0) {
        cml_free(dict);
        fclose(f);
        return NULL;
    }

    char* paren = strchr(sp, '(');
    int shape[NPY_MAX_DIMS];
    int ndim     = 0;
    size_t numel = 1;
    if (paren) {
        char* q = paren + 1;
        while (ndim < NPY_MAX_DIMS) {
            while (*q == ' ' || *q == ',')
                q++;
            if (*q == ')' || *q == '\0')
                break;
            int v         = atoi(q);
            shape[ndim++] = v;
            numel *= (size_t)v;
            while (*q && *q != ',' && *q != ')')
                q++;
        }
    }
    if (ndim == 0) {
        shape[ndim++] = 1; /* 0-d scalar -> 1-element vector */
        numel         = 1;
    }
    cml_free(dict);

    size_t nbytes = numel * cml_dtype_size(dtype);
    void* buf     = cml_malloc(nbytes ? nbytes : 1);
    if (!buf || fread(buf, 1, nbytes, f) != nbytes) {
        cml_free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    TensorConfig config = {.dtype = dtype, .has_dtype = true};
    Tensor* t           = tensor_from_data(buf, shape, ndim, &config);
    cml_free(buf);
    return t;
}
