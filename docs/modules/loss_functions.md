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
