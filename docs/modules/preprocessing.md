# Preprocessing

## Standard Scaler
- **Description**: Scales data to have zero mean and unit variance.
- **Functions**:
  - `float *standard_scaler(float *x, int size)`
- **File**: [`standard_scaler.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/standard_scaler.c)

## Min-Max Scaler
- **Description**: Scales data to a specified range (default: [0, 1]).
- **Functions**:
  - `float *min_max_scaler(float *x, int size)`
- **File**: [`min_max_scaler.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/min_max_scaler.c)

## Label Encoder
- **Description**: Encodes categorical labels as integers.
- **Functions**:
  - `int *label_encoder(char *x, int size, CharMap **map, int *mapSize)`
  - `char *label_decoder(int *x, int size, CharMap *map, int mapSize)`
  - `void free_label_memory(CharMap *map, int *encoded, char *decoded)`
- **File**: [`label_encoder.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/label_encoder.c)

## One-Hot Encoder
- **Description**: Encodes categorical labels as one-hot vectors.
- **Functions**:
  - `int *one_hot_encoding(char *x, int size, CharMap **map, int *mapSize)`
  - `char *one_hot_decoding(int *x, int size, CharMap *map, int mapSize)`
  - `void free_one_hot_memory(int *x, char *y, CharMap *map)`
- **File**: [`one_hot_encoder.c`](https://github.com/jaywyawhare/C-ML/tree/master/src/Preprocessing/one_hot_encoder.c)
