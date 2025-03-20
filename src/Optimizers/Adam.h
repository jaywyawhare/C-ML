#ifndef ADAM_H
#define ADAM_H

float Adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon);

#endif
