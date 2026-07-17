/* FFT: radix-2 (power-of-two) + DFT fallback, forward/inverse round-trip. */
#include <stdio.h>
#include <math.h>
#include "cml.h"

static int g_fail = 0;
#define CHECK(name, cond) do { \
    if (cond) printf("  PASS  %s\n", name); \
    else { printf("  FAIL  %s\n", name); g_fail = 1; } } while (0)

int main(void) {
    cml_init();
    printf("=== FFT ===\n");

    /* FFT of an impulse [1,0,0,...] -> all ones (real=1, imag=0) */
    { int n = 8; float re[8] = {1,0,0,0,0,0,0,0}, im[8] = {0};
      cml_fft_1d(re, im, n, 0);
      int ok = 1; for (int k = 0; k < n; k++) if (fabsf(re[k]-1.0f) > 1e-4f || fabsf(im[k]) > 1e-4f) ok = 0;
      CHECK("FFT(impulse) = all ones (pow2 radix-2)", ok); }

    /* forward then inverse = identity (pow2) */
    { int n = 16; float re[16], im[16], re0[16], im0[16];
      for (int i = 0; i < n; i++) { re[i] = sinf(0.5f*i) + 0.3f*i; im[i] = cosf(0.2f*i); re0[i]=re[i]; im0[i]=im[i]; }
      cml_fft_1d(re, im, n, 0);
      cml_fft_1d(re, im, n, 1);
      int ok = 1; for (int i = 0; i < n; i++) if (fabsf(re[i]-re0[i]) > 1e-3f || fabsf(im[i]-im0[i]) > 1e-3f) ok = 0;
      CHECK("IFFT(FFT(x)) == x (pow2)", ok); }

    /* non-power-of-two round-trip via DFT fallback */
    { int n = 6; float re[6], im[6], re0[6], im0[6];
      for (int i = 0; i < n; i++) { re[i] = (float)(i-2); im[i] = (float)(i%3); re0[i]=re[i]; im0[i]=im[i]; }
      cml_fft_1d(re, im, n, 0);
      cml_fft_1d(re, im, n, 1);
      int ok = 1; for (int i = 0; i < n; i++) if (fabsf(re[i]-re0[i]) > 1e-3f || fabsf(im[i]-im0[i]) > 1e-3f) ok = 0;
      CHECK("IFFT(FFT(x)) == x (non-pow2 DFT fallback)", ok); }

    /* known DC: FFT of constant [c,c,c,c] -> [4c, 0, 0, 0] */
    { int n = 4; float re[4] = {2,2,2,2}, im[4] = {0};
      cml_fft_1d(re, im, n, 0);
      CHECK("FFT(constant) DC bin = n*c", fabsf(re[0]-8.0f) < 1e-4f && fabsf(re[1]) < 1e-4f); }

    /* tensor API: [n,2] complex round-trip */
    { float xd[8] = {1,0, 2,0, 3,0, 4,0};   /* [4,2] real signal */
      Tensor* x = cml_tensor_2d(xd, 4, 2);
      Tensor* X = cml_fft(x, 0);
      Tensor* xi = cml_fft(X, 1);
      float* d = (float*)tensor_data_ptr(xi);
      int ok = X && xi && fabsf(d[0]-1.0f) < 1e-3f && fabsf(d[2]-2.0f) < 1e-3f && fabsf(d[6]-4.0f) < 1e-3f;
      CHECK("cml_fft tensor round-trip [n,2]", ok); }

    printf("\n%s\n", g_fail ? "FFT TESTS FAILED" : "All FFT tests passed");
    return g_fail;
}
