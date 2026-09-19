/* FFT: radix-2 (power-of-two) + DFT fallback, forward/inverse round-trip. */
#include <stdio.h>
#include <math.h>
#include "cml.h"
#include "test_harness.h"

int main(void) {
    cml_init();
    printf("=== FFT ===\n");

    /* FFT of an impulse [1,0,0,...] -> all ones (real=1, imag=0) */
    {
        int n       = 8;
        float re[8] = {1, 0, 0, 0, 0, 0, 0, 0}, im[8] = {0};
        cml_fft_1d(re, im, n, 0);
        int ok = 1;
        for (int k = 0; k < n; k++)
            if (fabsf(re[k] - 1.0f) > 1e-4f || fabsf(im[k]) > 1e-4f)
                ok = 0;
        CHECK("FFT(impulse) = all ones (pow2 radix-2)", ok);
    }

    /* forward then inverse = identity (pow2) */
    {
        int n = 16;
        float re[16], im[16], re0[16], im0[16];
        for (int i = 0; i < n; i++) {
            re[i]  = sinf(0.5f * i) + 0.3f * i;
            im[i]  = cosf(0.2f * i);
            re0[i] = re[i];
            im0[i] = im[i];
        }
        cml_fft_1d(re, im, n, 0);
        cml_fft_1d(re, im, n, 1);
        int ok = 1;
        for (int i = 0; i < n; i++)
            if (fabsf(re[i] - re0[i]) > 1e-3f || fabsf(im[i] - im0[i]) > 1e-3f)
                ok = 0;
        CHECK("IFFT(FFT(x)) == x (pow2)", ok);
    }

    /* non-power-of-two round-trip via DFT fallback */
    {
        int n = 6;
        float re[6], im[6], re0[6], im0[6];
        for (int i = 0; i < n; i++) {
            re[i]  = (float)(i - 2);
            im[i]  = (float)(i % 3);
            re0[i] = re[i];
            im0[i] = im[i];
        }
        cml_fft_1d(re, im, n, 0);
        cml_fft_1d(re, im, n, 1);
        int ok = 1;
        for (int i = 0; i < n; i++)
            if (fabsf(re[i] - re0[i]) > 1e-3f || fabsf(im[i] - im0[i]) > 1e-3f)
                ok = 0;
        CHECK("IFFT(FFT(x)) == x (non-pow2 DFT fallback)", ok);
    }

    /* known DC: FFT of constant [c,c,c,c] -> [4c, 0, 0, 0] */
    {
        int n       = 4;
        float re[4] = {2, 2, 2, 2}, im[4] = {0};
        cml_fft_1d(re, im, n, 0);
        CHECK("FFT(constant) DC bin = n*c", fabsf(re[0] - 8.0f) < 1e-4f && fabsf(re[1]) < 1e-4f);
    }

    /* tensor API: [n,2] complex round-trip */
    {
        float xd[8] = {1, 0, 2, 0, 3, 0, 4, 0}; /* [4,2] real signal */
        Tensor* x   = cml_tensor_2d(xd, 4, 2);
        Tensor* X   = cml_fft(x, 0);
        Tensor* xi  = cml_fft(X, 1);
        float* d    = (float*)tensor_data_ptr(xi);
        int ok      = X && xi && fabsf(d[0] - 1.0f) < 1e-3f && fabsf(d[2] - 2.0f) < 1e-3f &&
                 fabsf(d[6] - 4.0f) < 1e-3f;
        CHECK("cml_fft tensor round-trip [n,2]", ok);
    }

    /* 2-D FFT round-trip on a [4,4,2] complex image */
    {
        float img[32];
        for (int i = 0; i < 16; i++) {
            img[2 * i]     = (float)((i * 7) % 5) - 2.0f;
            img[2 * i + 1] = 0.0f;
        }
        float orig[32];
        for (int i = 0; i < 32; i++)
            orig[i] = img[i];
        int shp[3] = {4, 4, 2};
        Tensor* x  = cml_tensor(img, shp, 3, NULL);
        Tensor* X  = cml_fft2(x, 0);
        Tensor* xi = cml_fft2(X, 1);
        float* d   = (float*)tensor_data_ptr(xi);
        int ok     = X && xi;
        for (int i = 0; i < 32 && ok; i++)
            if (fabsf(d[i] - orig[i]) > 1e-3f)
                ok = 0;
        CHECK("cml_fft2 2-D round-trip [4,4,2]", ok);
    }

    return TEST_SUMMARY();
}
