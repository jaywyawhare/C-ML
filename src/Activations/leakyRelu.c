#include<math.h>

float leakyRelu(float x) {
    if (x > 0) return x;
    else return 0.01 * x;
}
