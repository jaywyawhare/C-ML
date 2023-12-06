#include<math.h>

float leakyRelu(float x) {
   return x > 0 ? x : 0.01 * x;
}
