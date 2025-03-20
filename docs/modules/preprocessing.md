# Preprocessing


## Standard Scaler
- **Description**: Scales data to have zero mean and unit variance.
- **Function**: `standardScaler(float *x, int size)`
- **File**: [`standardScaler.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/standardScaler.c)

## Min-Max Scaler
- **Description**: Scales data to a specified range (default: [0, 1]).
- **Function**: `minMaxScaler(float *x, int size)`
- **File**: [`minMaxScaler.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/minMaxScaler.c)

## Label Encoder
- **Description**: Encodes categorical labels as integers.
- **Function**: `labelEncoder(char *x, int size, ...)`
- **File**: [`labelEncoder.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/labelEncoder.c)

## One-Hot Encoder
- **Description**: Encodes categorical labels as one-hot vectors.
- **Function**: `oneHotEncoding(char *x, int size, ...)`
- **File**: [`oneHotEncoder.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/oneHotEncoder.c)
