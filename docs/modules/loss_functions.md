# Loss Functions

## Mean Squared Error
- **Description**: Measures the average squared difference between predictions and actual values.
- **Function**: `mean_squared_error(float *y, float *yHat, int n)`
- **File**: [`mean_squared_error.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/mean_squared_error.c)

## Binary Cross-Entropy
- **Description**: Loss function for binary classification tasks.
- **Function**: `binary_cross_entropy_loss(float *yHat, float *y, int size)`
- **File**: [`binary_cross_entropy_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/binary_cross_entropy_loss.c)

## Focal Loss
- **Description**: Focuses on hard-to-classify examples by down-weighting easy examples.
- **Function**: `focal_loss(float *y, float *yHat, int n, float gamma)`
- **File**: [`focal_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/focal_loss.c)

## Mean Absolute Error
- **Description**: Measures the average absolute difference between predictions and actual values.
- **Function**: `mean_absolute_error(float *y, float *yHat, int n)`
- **File**: [`mean_absolute_error.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/mean_absolute_error.c)

## Mean Absolute Percentage Error
- **Description**: Measures the percentage difference between predictions and actual values.
- **Function**: `mean_absolute_percentage_error(float *y, float *yHat, int n)`
- **File**: [`mean_absolute_percentage_error.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/mean_absolute_percentage_error.c)

## Root Mean Squared Error
- **Description**: Square root of the mean squared error.
- **Function**: `root_mean_squared_error(float *y, float *yHat, int n)`
- **File**: [`root_mean_squared_error.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/root_mean_squared_error.c)

## Reduce Mean
- **Description**: Computes the mean of an array of values.
- **Function**: `reduce_mean(float *loss, int size)`
- **File**: [`reduce_mean.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/reduce_mean.c)

## Huber Loss
- **Description**: Huber Loss function.
- **Function**: `huber_loss(float *y, float *yHat, int n)`
- **File**: [`huber_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/huber_loss.c)

## KLD Loss
- **Description**: Kullback-Leibler Divergence Loss function.
- **Function**: `kld_loss(float *p, float *q, int n)`
- **File**: [`kld_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/kld_loss.c)

## Log-Cosh Loss
- **Description**: Log-Cosh Loss function.
- **Function**: `log_cosh_loss(float *y, float *yHat, int n)`
- **File**: [`log_cosh_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/log_cosh_loss.c)

## Poisson Loss
- **Description**: Poisson Loss function.
- **Function**: `poisson_loss(float *y, float *yHat, int n)`
- **File**: [`poisson_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/poisson_loss.c)

## Smooth L1 Loss
- **Description**: Smooth L1 Loss function.
- **Function**: `smooth_l1_loss(float *y, float *yHat, int n)`
- **File**: [`smooth_l1_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/smooth_l1_loss.c)

## Tversky Loss
- **Description**: Tversky Loss function.
- **Function**: `tversky_loss(float *y, float *yHat, int n)`
- **File**: [`tversky_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/tversky_loss.c)

## Cosine Similarity Loss
- **Description**: Cosine Similarity Loss function.
- **Function**: `cosine_similarity_loss(float *y, float *yHat, int n)`
- **File**: [`cosine_similarity_loss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/cosine_similarity_loss.c)
