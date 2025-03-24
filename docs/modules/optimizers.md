# Optimizers

## SGD (Stochastic Gradient Descent)
- **Description**: Basic optimizer that updates weights using gradients.
- **Function**: `float sgd(float x, float y, float lr, float *w, float *b)`
- **File**: [`sgd.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Optimizers/sgd.c)

## Adam
- **Description**: Adaptive optimizer combining momentum and RMSprop.
- **Function**: `float adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon)`
- **File**: [`Adam.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Optimizers/adam.c)

## RMSprop
- **Description**: Optimizer that scales learning rates based on recent gradients.
- **Function**: `float rms_prop(float x, float y, float lr, float *w, float *b, float *cache_w, float *cache_b, float epsilon, float beta)`
- **File**: [`RMSprop.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Optimizers/rmsprop.c)
