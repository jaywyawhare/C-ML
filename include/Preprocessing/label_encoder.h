#ifndef LABEL_ENCODER_H
#define LABEL_ENCODER_H

typedef struct
{
    char character;
    int encodedValue;
} CharMap;

/**
 * @brief Maps characters to unique integer labels.
 *
 * @param x The input character array.
 * @param size The size of the input array.
 * @param map A pointer to the character-to-integer mapping.
 * @param mapSize A pointer to store the size of the mapping.
 * @return A pointer to the encoded integer array.
 */
int *label_encoder(char *x, int size, CharMap **map, int *mapSize);

/**
 * @brief Decodes integer labels back into characters.
 *
 * @param x The encoded integer array.
 * @param size The size of the input array.
 * @param map The character-to-integer mapping.
 * @param mapSize The size of the mapping.
 * @return A pointer to the decoded character array.
 */
char *label_decoder(int *x, int size, CharMap *map, int mapSize);

/**
 * @brief Frees the memory allocated for label encoding and decoding.
 *
 * @param map The character-to-integer mapping.
 * @param encoded The encoded integer array.
 * @param decoded The decoded character array.
 */
void free_label_memory(CharMap *map, int *encoded, char *decoded);

#endif
