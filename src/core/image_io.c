#include "core/image_io.h"
#include "core/logging.h"
#include "alloc/cml_allocator.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

/* ---- DEFLATE inflate (RFC 1951), canonical-Huffman "puff" style ---------- */

#define INFLATE_MAXBITS 15
#define INFLATE_MAXLCODES 286
#define INFLATE_MAXDCODES 30
#define INFLATE_MAXCODES (INFLATE_MAXLCODES + INFLATE_MAXDCODES)

typedef struct {
    const unsigned char* in;
    size_t inlen;
    size_t incnt;
    int bitbuf;
    int bitcnt;
    unsigned char* out;
    size_t outlen;
    size_t outcap;
} Inflater;

typedef struct {
    short count[INFLATE_MAXBITS + 1];
    short symbol[INFLATE_MAXCODES];
} Huff;

/** Read one LSB-first bit from the stream; -1 past the end.
 * @param s (Inflater*)
 * @return (int) */
static int inf_bit(Inflater* s) {
    if (s->bitcnt == 0) {
        if (s->incnt >= s->inlen)
            return -1;
        s->bitbuf = s->in[s->incnt++];
        s->bitcnt = 8;
    }
    int b = s->bitbuf & 1;
    s->bitbuf >>= 1;
    s->bitcnt--;
    return b;
}

/** Read @p need LSB-first bits as an unsigned value; -1 past the end.
 * @param s (Inflater*)
 * @param need (int)
 * @return (int) */
static int inf_bits(Inflater* s, int need) {
    int val = 0;
    for (int i = 0; i < need; i++) {
        int b = inf_bit(s);
        if (b < 0)
            return -1;
        val |= b << i;
    }
    return val;
}

/** Append one byte to the output buffer, growing it as needed. Returns 0 on success.
 * @param s (Inflater*)
 * @param c (unsigned char)
 * @return (int) */
static int inf_put(Inflater* s, unsigned char c) {
    if (s->outlen >= s->outcap) {
        size_t nc           = s->outcap ? s->outcap * 2 : 256;
        unsigned char* nout = cml_realloc(s->out, nc);
        if (!nout)
            return -1;
        s->out    = nout;
        s->outcap = nc;
    }
    s->out[s->outlen++] = c;
    return 0;
}

/** Decode one canonical-Huffman symbol from the stream; -1 on error.
 * @param s (Inflater*)
 * @param h (const Huff*)
 * @return (int) */
static int inf_decode(Inflater* s, const Huff* h) {
    int code = 0, first = 0, index = 0;
    for (int len = 1; len <= INFLATE_MAXBITS; len++) {
        int b = inf_bit(s);
        if (b < 0)
            return -1;
        code |= b;
        int count = h->count[len];
        if (code - first < count)
            return h->symbol[index + (code - first)];
        index += count;
        first += count;
        first <<= 1;
        code <<= 1;
    }
    return -1;
}

/** Build a canonical Huffman table from per-symbol code lengths.
 * @param h (Huff*)
 * @param lengths (const short*)
 * @param n (int)
 * @return (void) */
static void inf_construct(Huff* h, const short* lengths, int n) {
    for (int len = 0; len <= INFLATE_MAXBITS; len++)
        h->count[len] = 0;
    for (int sym = 0; sym < n; sym++)
        h->count[lengths[sym]]++;
    h->count[0] = 0;
    short offs[INFLATE_MAXBITS + 1];
    offs[1] = 0;
    for (int len = 1; len < INFLATE_MAXBITS; len++)
        offs[len + 1] = offs[len] + h->count[len];
    for (int sym = 0; sym < n; sym++)
        if (lengths[sym])
            h->symbol[offs[lengths[sym]]++] = (short)sym;
}

static const short LEN_BASE[29]  = {3,  4,  5,  6,  7,  8,  9,  10, 11,  13,  15,  17,  19,  23, 27,
                                    31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
static const short LEN_EXTRA[29] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2,
                                    2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
static const short DIST_BASE[30] = {1,    2,    3,    4,    5,    7,    9,    13,    17,    25,
                                    33,   49,   65,   97,   129,  193,  257,  385,   513,   769,
                                    1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
static const short DIST_EXTRA[30] = {0, 0, 0, 0, 1, 1, 2, 2,  3,  3,  4,  4,  5,  5,  6,
                                     6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};

/** Decode one block body given its literal/length and distance tables. Returns 0 ok.
 * @param s (Inflater*)
 * @param lh (const Huff*)
 * @param dh (const Huff*)
 * @return (int) */
static int inf_codes(Inflater* s, const Huff* lh, const Huff* dh) {
    for (;;) {
        int sym = inf_decode(s, lh);
        if (sym < 0)
            return -1;
        if (sym == 256)
            return 0; /* end of block */
        if (sym < 256) {
            if (inf_put(s, (unsigned char)sym) != 0)
                return -1;
            continue;
        }
        sym -= 257;
        if (sym >= 29)
            return -1;
        int len  = LEN_BASE[sym] + inf_bits(s, LEN_EXTRA[sym]);
        int dsym = inf_decode(s, dh);
        if (dsym < 0 || dsym >= 30)
            return -1;
        int dist = DIST_BASE[dsym] + inf_bits(s, DIST_EXTRA[dsym]);
        if ((size_t)dist > s->outlen)
            return -1;
        for (int i = 0; i < len; i++)
            if (inf_put(s, s->out[s->outlen - (size_t)dist]) != 0)
                return -1;
    }
}

/** Build the fixed Huffman tables (RFC 1951 3.2.6) and decode the block.
 * @param s (Inflater*)
 * @return (int) */
static int inf_fixed(Inflater* s) {
    short ll[288], dl[30];
    for (int i = 0; i < 144; i++)
        ll[i] = 8;
    for (int i = 144; i < 256; i++)
        ll[i] = 9;
    for (int i = 256; i < 280; i++)
        ll[i] = 7;
    for (int i = 280; i < 288; i++)
        ll[i] = 8;
    for (int i = 0; i < 30; i++)
        dl[i] = 5;
    Huff lh, dh;
    inf_construct(&lh, ll, 288);
    inf_construct(&dh, dl, 30);
    return inf_codes(s, &lh, &dh);
}

/** Read the dynamic Huffman tables and decode the block (RFC 1951 3.2.7).
 * @param s (Inflater*)
 * @return (int) */
static int inf_dynamic(Inflater* s) {
    static const short ORDER[19] = {16, 17, 18, 0, 8,  7, 9,  6, 10, 5,
                                    11, 4,  12, 3, 13, 2, 14, 1, 15};
    int hlit                     = inf_bits(s, 5);
    int hdist                    = inf_bits(s, 5);
    int hclen                    = inf_bits(s, 4);
    if (hlit < 0 || hdist < 0 || hclen < 0)
        return -1;
    hlit += 257;
    hdist += 1;
    hclen += 4;
    if (hlit > INFLATE_MAXLCODES || hdist > INFLATE_MAXDCODES)
        return -1;

    short cl[19] = {0};
    for (int i = 0; i < hclen; i++) {
        int v = inf_bits(s, 3);
        if (v < 0)
            return -1;
        cl[ORDER[i]] = (short)v;
    }
    Huff clh;
    inf_construct(&clh, cl, 19);

    short lengths[INFLATE_MAXCODES] = {0};
    int n                           = 0;
    while (n < hlit + hdist) {
        int sym = inf_decode(s, &clh);
        if (sym < 0)
            return -1;
        if (sym < 16) {
            lengths[n++] = (short)sym;
        } else if (sym == 16) {
            if (n == 0)
                return -1;
            int rep = 3 + inf_bits(s, 2);
            short v = lengths[n - 1];
            while (rep-- > 0 && n < hlit + hdist)
                lengths[n++] = v;
        } else if (sym == 17) {
            int rep = 3 + inf_bits(s, 3);
            while (rep-- > 0 && n < hlit + hdist)
                lengths[n++] = 0;
        } else { /* 18 */
            int rep = 11 + inf_bits(s, 7);
            while (rep-- > 0 && n < hlit + hdist)
                lengths[n++] = 0;
        }
    }
    Huff lh, dh;
    inf_construct(&lh, lengths, hlit);
    inf_construct(&dh, lengths + hlit, hdist);
    return inf_codes(s, &lh, &dh);
}

/** Inflate a raw DEFLATE stream.
 * @param data (const unsigned char*)
 * @param len (size_t)
 * @param out_len (size_t*)
 * @return (unsigned char*)
 */
unsigned char* cml_inflate(const unsigned char* data, size_t len, size_t* out_len) {
    if (!data || !out_len)
        return NULL;
    Inflater s = {.in = data, .inlen = len};
    int final  = 0;
    do {
        final     = inf_bit(&s);
        int btype = inf_bits(&s, 2);
        if (final < 0 || btype < 0) {
            cml_free(s.out);
            return NULL;
        }
        int rc;
        if (btype == 0) { /* stored */
            s.bitbuf = 0;
            s.bitcnt = 0; /* align to byte boundary */
            if (s.incnt + 4 > s.inlen) {
                cml_free(s.out);
                return NULL;
            }
            int blen = s.in[s.incnt] | (s.in[s.incnt + 1] << 8);
            s.incnt += 4; /* skip LEN + NLEN */
            if (s.incnt + (size_t)blen > s.inlen) {
                cml_free(s.out);
                return NULL;
            }
            rc = 0;
            for (int i = 0; i < blen && rc == 0; i++)
                rc = inf_put(&s, s.in[s.incnt++]);
        } else if (btype == 1) {
            rc = inf_fixed(&s);
        } else if (btype == 2) {
            rc = inf_dynamic(&s);
        } else {
            rc = -1;
        }
        if (rc != 0) {
            cml_free(s.out);
            return NULL;
        }
    } while (!final);
    *out_len = s.outlen;
    return s.out ? s.out : (unsigned char*)cml_calloc(1, 1);
}

/** Inflate a zlib stream (RFC 1950 wrapper around DEFLATE).
 * @param data (const unsigned char*)
 * @param len (size_t)
 * @param out_len (size_t*)
 * @return (unsigned char*)
 */
unsigned char* cml_zlib_inflate(const unsigned char* data, size_t len, size_t* out_len) {
    if (!data || len < 6)
        return NULL;
    /* 2-byte header (CMF/FLG); FDICT must be clear. 4-byte Adler trailer ignored. */
    if ((data[0] & 0x0f) != 8 || (data[1] & 0x20))
        return NULL;
    return cml_inflate(data + 2, len - 2 - 4, out_len);
}

/* ---- PNG decode (8-bit, no interlace/palette) ---------------------------- */

static uint32_t be32(const unsigned char* p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | p[3];
}

static int paeth(int a, int b, int c) {
    int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
    if (pa <= pb && pa <= pc)
        return a;
    return pb <= pc ? b : c;
}

/** Load an 8-bit PNG into an HWC float buffer in [0,1].
 * @param path (const char*)
 * @param width (int*)
 * @param height (int*)
 * @param channels (int*)
 * @return (float*)
 */
float* cml_image_load_png(const char* path, int* width, int* height, int* channels) {
    if (!path || !width || !height || !channels)
        return NULL;
    FILE* f = fopen(path, "rb");
    if (!f) {
        LOG_ERROR("cml_image_load_png: cannot open %s", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long fsz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (fsz < 8) {
        fclose(f);
        return NULL;
    }
    unsigned char* buf = cml_malloc((size_t)fsz);
    if (!buf || fread(buf, 1, (size_t)fsz, f) != (size_t)fsz) {
        fclose(f);
        cml_free(buf);
        return NULL;
    }
    fclose(f);

    static const unsigned char SIG[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    if (memcmp(buf, SIG, 8) != 0) {
        LOG_ERROR("cml_image_load_png: bad signature");
        cml_free(buf);
        return NULL;
    }

    int w = 0, h = 0, ch = 0;
    unsigned char* idat = NULL;
    size_t idat_len     = 0;
    size_t pos          = 8;
    int ok              = 1;
    while (pos + 8 <= (size_t)fsz) {
        uint32_t clen            = be32(buf + pos);
        const unsigned char* typ = buf + pos + 4;
        const unsigned char* dat = buf + pos + 8;
        if (pos + 12 + clen > (size_t)fsz)
            break;
        if (memcmp(typ, "IHDR", 4) == 0) {
            w             = (int)be32(dat);
            h             = (int)be32(dat + 4);
            int bitdepth  = dat[8];
            int colortype = dat[9];
            int interlace = dat[12];
            if (bitdepth != 8 || interlace != 0) {
                LOG_ERROR("cml_image_load_png: only 8-bit non-interlaced PNGs supported");
                ok = 0;
                break;
            }
            switch (colortype) {
            case 0:
                ch = 1;
                break; /* grayscale */
            case 2:
                ch = 3;
                break; /* RGB */
            case 4:
                ch = 2;
                break; /* gray + alpha */
            case 6:
                ch = 4;
                break; /* RGBA */
            default:
                LOG_ERROR("cml_image_load_png: palette/unsupported color type %d", colortype);
                ok = 0;
            }
            if (!ok)
                break;
        } else if (memcmp(typ, "IDAT", 4) == 0) {
            unsigned char* n = cml_realloc(idat, idat_len + clen);
            if (!n) {
                ok = 0;
                break;
            }
            idat = n;
            memcpy(idat + idat_len, dat, clen);
            idat_len += clen;
        } else if (memcmp(typ, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + clen; /* length + type + data + crc */
    }

    float* out = NULL;
    if (ok && w > 0 && h > 0 && ch > 0 && idat) {
        size_t raw_len     = 0;
        unsigned char* raw = cml_zlib_inflate(idat, idat_len, &raw_len);
        size_t stride      = (size_t)w * ch;
        if (raw && raw_len >= (size_t)h * (stride + 1)) {
            unsigned char* img = cml_malloc((size_t)h * stride);
            out                = cml_malloc((size_t)h * stride * sizeof(float));
            if (img && out) {
                /* Unfilter each scanline (filter byte precedes its row). */
                for (int y = 0; y < h; y++) {
                    const unsigned char* src = raw + (size_t)y * (stride + 1);
                    int ft                   = src[0];
                    unsigned char* row       = img + (size_t)y * stride;
                    unsigned char* prev      = y ? img + (size_t)(y - 1) * stride : NULL;
                    for (size_t x = 0; x < stride; x++) {
                        int rawv = src[1 + x];
                        int a    = x >= (size_t)ch ? row[x - ch] : 0;
                        int b    = prev ? prev[x] : 0;
                        int c    = (prev && x >= (size_t)ch) ? prev[x - ch] : 0;
                        int val;
                        switch (ft) {
                        case 0:
                            val = rawv;
                            break;
                        case 1:
                            val = rawv + a;
                            break;
                        case 2:
                            val = rawv + b;
                            break;
                        case 3:
                            val = rawv + ((a + b) >> 1);
                            break;
                        case 4:
                            val = rawv + paeth(a, b, c);
                            break;
                        default:
                            val = rawv;
                            break;
                        }
                        row[x] = (unsigned char)(val & 0xff);
                    }
                }
                for (size_t i = 0; i < (size_t)h * stride; i++)
                    out[i] = img[i] / 255.0f;
                *width    = w;
                *height   = h;
                *channels = ch;
            } else {
                cml_free(out);
                out = NULL;
            }
            cml_free(img);
        }
        cml_free(raw);
    }
    cml_free(idat);
    cml_free(buf);
    return out;
}
