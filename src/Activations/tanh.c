#include <math.h>
#include <float.h> 

float tanH(float x){
    if (x > 20.0f) { 
        return 1.0f;
    } else if (x < -20.0f) { 
        return -1.0f;
    } else {
        float e_pos = expf(x);
        float e_neg = expf(-x);
        return (e_pos - e_neg) / (e_pos + e_neg);
    }
}