#include <math.h>

float sigmoid(float x){
    return 1 / (1 + expf(-x));
}