#include <math.h>

float elu(float x, float alpha) {
    return x >= 0 ? x : alpha * (expf(x) - 1);
}
