#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../../include/Preprocessing/label_encoder.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"

#include "../../include/Core/logging.h"
#ifndef DEBUG_LOGGING
#define DEBUG_LOGGING 0
#endif

/**
 * @brief Encodes a character array into integer labels.
 *
 * The function maps each unique character in the input array to a unique integer label.
 *
 * @param x The input character array.
 * @param size The size of the input array.
 * @param map A pointer to the character-to-integer mapping.
 * @param mapSize A pointer to store the size of the mapping.
 * @return A pointer to the encoded integer array, or an error code.
 */
int *label_encoder(char *x, int size, CharMap **map, int *mapSize)
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
        LOG_ERROR("Memory allocation failed\n");
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

    int *encoded = (int *)cm_safe_malloc(sizeof(int) * size, __FILE__, __LINE__);
    if (encoded == NULL)
    {
        LOG_ERROR("Memory allocation failed\n");
        free(*map);
        return (int *)CM_MEMORY_ALLOCATION_ERROR;
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < *mapSize; j++)
        {
            if (x[i] == (*map)[j].character)
            {
                encoded[i] = (*map)[j].encodedValue;
                break;
            }
        }
    }
    LOG_DEBUG("Encoding complete.");
    return encoded;
}

/**
 * @brief Decodes integer labels back into a character array.
 *
 * The function converts the encoded integer labels back into the original character array
 * using the provided mapping.
 *
 * @param x The encoded integer array.
 * @param size The size of the input array.
 * @param map The character-to-integer mapping.
 * @param mapSize The size of the mapping.
 * @return A pointer to the decoded character array, or NULL if an error occurs.
 */
char *label_decoder(int *x, int size, CharMap *map, int mapSize)
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
        LOG_ERROR("Memory allocation failed\n");
        return NULL;
    }
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < mapSize; j++)
        {
            if (x[i] == map[j].encodedValue)
            {
                decoded[i] = map[j].character;
                break;
            }
        }
    }
    decoded[size] = '\0';
    LOG_DEBUG("Decoding complete.");
    return decoded;
}

/**
 * @brief Frees the memory allocated for label encoding and decoding.
 *
 * This function releases the memory allocated for the character-to-integer mapping,
 * the encoded integer array, and the decoded character array.
 *
 * @param map The character-to-integer mapping.
 * @param encoded The encoded integer array.
 * @param decoded The decoded character array.
 */
void free_label_memory(CharMap *map, int *encoded, char *decoded)
{
    cm_safe_free((void **)&map);
    cm_safe_free((void **)&encoded);
    cm_safe_free((void **)&decoded);
}