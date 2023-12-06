#include<math.h>

float elu(float x){
    float alpha = 1.69;
    return x > 0 ? x : alpha *(exp(x) - 1);
}
