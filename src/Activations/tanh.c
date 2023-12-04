#include <math.h>

float tanH(float x){
    return (float) (expf(2*x) - 1) / (expf(2*x) + 1);
}