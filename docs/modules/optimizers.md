# Optimizers


## SGD (Stochastic Gradient Descent)
- **Description**: Basic optimizer that updates weights using gradients.
- **Function**: `SGD(float x, float y, float lr, float *w, float *b)`
- **File**: [`sgd.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Optimizers/sgd.c)

## Adam
- **Description**: Adaptive optimizer combining momentum and RMSprop.
- **Function**: `Adam(float x, float y, float lr, float *w, float *b, ...)`
- **File**: [`Adam.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Optimizers/Adam.c)

## RMSprop
- **Description**: Optimizer that scales learning rates based on recent gradients.
- **Function**: `RMSprop(float x, float y, float lr, float *w, float *b, ...)`
- **File**: [`RMSprop.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Optimizers/RMSprop.c)
