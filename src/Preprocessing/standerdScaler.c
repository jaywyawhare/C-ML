#include<math.h>

float* standerdScaler(float *x){
    float *scaled = malloc(sizeof(float) * sizeof(x));
    float mean = 0;
    for(int i = 0; i < sizeof(x); i++){
        mean += x[i];
    }
    mean /= sizeof(x);
    float std = 0;
    for(int i = 0; i < sizeof(x); i++){
        std += pow(x[i] - mean, 2);
    }
    std /= sizeof(x);
    std = sqrt(std);
    for(int i = 0; i < sizeof(x); i++){
        scaled[i] = (x[i] - mean) / std;
    }
    return scaled;
}