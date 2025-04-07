#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "../../include/Preprocessing/one_hot_encoder.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Encodes a character array into a one-hot encoded integer array.
 *
 * The function maps each unique character in the input array to a unique integer label
 * and creates a one-hot encoded representation of the input array.
 *
 * @param x The input character array.
 * @param size The size of the input array.
 * @param map A pointer to the character-to-integer mapping.
 * @param mapSize A pointer to store the size of the mapping.
 * @return A pointer to the one-hot encoded array, or an error code.
 */
int *one_hot_encoding(char *x, int size, CharMap **map, int *mapSize)
{
    if (x == NULL || map == NULL || mapSize == NULL)
    {
        LOG_ERROR("Null pointer argument");
        return (int *)CM_NULL_POINTER_ERROR;
    }

    if (size <= 0)
    {
        LOG_ERROR("Invalid size argument");
        return (int *)CM_INVALID_PARAMETER_ERROR;
    }

    *map = (CharMap *)cm_safe_malloc(sizeof(CharMap) * size, __FILE__, __LINE__);
    if (*map == NULL)
    {
        fprintf(stderr, "[oneHotEncoding] Memory allocation failed\n");
        return (int *)CM_MEMORY_ALLOCATION_ERROR;
    }

    int uniqueCount = 0;
    for (int i = 0; i < size; i++)
    {
        int isNew = 1;
        for (int j = 0; j < uniqueCount; j++)
        {
            if (x[i] == (*map)[j].character)
            {
                isNew = 0;
                break;
            }
        }
        if (isNew)
        {
            (*map)[uniqueCount].character = x[i];
            (*map)[uniqueCount].encodedValue = uniqueCount;
            uniqueCount++;
        }
    }
    *mapSize = uniqueCount;

    int *encoded = (int *)cm_safe_malloc(sizeof(int) * size * uniqueCount, __FILE__, __LINE__);
    if (encoded == NULL)
    {
        fprintf(stderr, "[oneHotEncoding] Memory allocation failed\n");
        free(*map);
        *map = NULL; 
        return (int *)CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < uniqueCount; j++)
        {
            encoded[i * uniqueCount + j] = (x[i] == (*map)[j].character) ? 1 : 0;
#if DEBUG_LOGGING
            LOG_DEBUG("encoded[%d]: %d", i * uniqueCount + j, encoded[i * uniqueCount + j]);
#endif
        }
    }
#if DEBUG_LOGGING
    printf("[oneHotEncoding] Debug: Encoding complete.\n");
#endif
    return encoded;
}

/**
 * @brief Decodes a one-hot encoded integer array back into a character array.
 *
 * The function converts a one-hot encoded representation back into the original
 * character array using the provided mapping.
 *
 * @param x The one-hot encoded integer array.
 * @param size The size of the input array.
 * @param map The character-to-integer mapping.
 * @param mapSize The size of the mapping.
 * @return A pointer to the decoded character array, or NULL if an error occurs.
 */
char *one_hot_decoding(int *x, int size, CharMap *map, int mapSize)
{
    if (x == NULL || map == NULL)
    {
        LOG_ERROR("Null pointer argument");
        return NULL;
    }

    if (size <= 0 || mapSize <= 0)
    {
        LOG_ERROR("Invalid size argument");
        return NULL;
    }
    char *decoded = (char *)cm_safe_malloc(sizeof(char) * (size + 1), __FILE__, __LINE__);
    if (decoded == NULL)
    {
        fprintf(stderr, "[oneHotEncoding] Memory allocation failed\n");
        return NULL;
    }
    for (int i = 0; i < size; i++)
    {
        decoded[i] = '\0';
        for (int j = 0; j < mapSize; j++)
        {
            if (x[i * mapSize + j] == 1)
            {
                decoded[i] = map[j].character;
                break;
            }
        }
    }
    decoded[size] = '\0';
#if DEBUG_LOGGING
    printf("[oneHotEncoding] Debug: Decoding complete.\n");
#endif
    return decoded;
}

/**
 * @brief Frees the memory allocated for one-hot encoding and decoding.
 *
 * This function releases the memory allocated for the one-hot encoded array,
 * the decoded character array, and the character-to-integer mapping.
 *
 * @param x The one-hot encoded integer array.
 * @param y The decoded character array.
 * @param map The character-to-integer mapping.
 */
void free_one_hot_memory(int *x, char *y, CharMap *map)
{
    cm_safe_free((void **)&x);
    cm_safe_free((void **)&y);
    cm_safe_free((void **)&map);
}