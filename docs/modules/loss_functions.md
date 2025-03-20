# Loss Functions


## Mean Squared Error
- **Description**: Measures the average squared difference between predictions and actual values.
- **Function**: `meanSquaredError(float *y, float *yHat, int n)`
- **File**: [`meanSquaredError.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/meanSquaredError.c)

## Binary Cross-Entropy
- **Description**: Loss function for binary classification tasks.
- **Function**: `binaryCrossEntropyLoss(float *yHat, float *y, int size)`
- **File**: [`binaryCrossEntropyLoss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/binaryCrossEntropyLoss.c)

## Focal Loss
- **Description**: Focuses on hard-to-classify examples by down-weighting easy examples.
- **Function**: `focalLoss(float *y, float *yHat, int n, float gamma)`
- **File**: [`focalLoss.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/focalLoss.c)

## Mean Absolute Error
- **Description**: Measures the average absolute difference between predictions and actual values.
- **Function**: `meanAbsoluteError(float *y, float *yHat, int n)`
- **File**: [`meanAbsoluteError.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/meanAbsoluteError.c)

## Mean Absolute Percentage Error
- **Description**: Measures the percentage difference between predictions and actual values.
- **Function**: `meanAbsolutePercentageError(float *y, float *yHat, int n)`
- **File**: [`meanAbsolutePercentageError.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/meanAbsolutePercentageError.c)

## Root Mean Squared Error
- **Description**: Square root of the mean squared error.
- **Function**: `rootMeanSquaredError(float *y, float *yHat, int n)`
- **File**: [`rootMeanSquaredError.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/rootMeanSquaredError.c)

## Reduce Mean
- **Description**: Computes the mean of an array of values.
- **Function**: `reduceMean(float *loss, int size)`
- **File**: [`reduceMean.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Loss_Functions/reduceMean.c)
