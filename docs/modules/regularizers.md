# Regularizers

## L1 Regularization
- **Description**: Adds the absolute value of weights to the loss function to encourage sparsity.
- **Function**: `l1(float x, float y, ...)`
- **File**: [`l1.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Regularizers/l1.c)

## L2 Regularization
- **Description**: Adds the squared value of weights to the loss function to prevent overfitting.
- **Function**: `l2(float x, float y, ...)`
- **File**: [`l2.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Regularizers/l2.c)

## Combined L1-L2 Regularization
- **Description**: Combines L1 and L2 regularization techniques.
- **Function**: `l1_l2(float *w, float *dw, float l1, float l2, int n)`
- **File**: [`l1_l2.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Regularizers/l1_l2.c)
