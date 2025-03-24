#ifndef ONE_HOT_ENCODER_H
#define ONE_HOT_ENCODER_H

/**
 * @brief Maps characters to unique integer labels for one-hot encoding.
 */
typedef struct
{
    char character;
    int encodedValue;
} CharMap;

/**
 * @brief Encodes a character array into a one-hot encoded integer array.
 *
 * @param x The input character array.
 * @param size The size of the input array.
 * @param map A pointer to the character-to-integer mapping.
 * @param mapSize A pointer to store the size of the mapping.
 * @return A pointer to the one-hot encoded array, or an error code.
 */
int *one_hot_encoding(char *x, int size, CharMap **map, int *mapSize);

/**
 * @brief Decodes a one-hot encoded integer array back into a character array.
 *
 * @param x The one-hot encoded integer array.
 * @param size The size of the input array.
 * @param map The character-to-integer mapping.
 * @param mapSize The size of the mapping.
 * @return A pointer to the decoded character array, or NULL if an error occurs.
 */
char *one_hot_decoding(int *x, int size, CharMap *map, int mapSize);

/**
 * @brief Frees the memory allocated for one-hot encoding and decoding.
 *
 * @param x The one-hot encoded integer array.
 * @param y The decoded character array.
 * @param map The character-to-integer mapping.
 */
void free_one_hot_memory(int *x, char *y, CharMap *map);

#endif
