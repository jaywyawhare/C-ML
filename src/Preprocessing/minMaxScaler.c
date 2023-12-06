#include<math.h>

float* minMaxScaler(float *x){
    float *scaled = malloc(sizeof(float) * sizeof(x));
    float min = x[0];
    float max = x[0];
    for(int i = 0; i < sizeof(x); i++){
        if(x[i] < min){
            min = x[i];
        }
        if(x[i] > max){
            max = x[i];
        }
    }
    for(int i = 0; i < sizeof(x); i++){
        scaled[i] = (x[i] - min) / (max - min);
    }
    return scaled;
}