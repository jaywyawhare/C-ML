#include <string.h>
#include "../../include/Preprocessing/one_hot_encoder.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

Node *one_hot_encoding_tensor(char *x, int size, CharMap **map, int *mapSize)
{
    if (!x || !map || !mapSize)
    {
        LOG_ERROR("Null pointer argument");
        return NULL;
    }

    // Create and build character map
    *map = (CharMap *)cm_safe_malloc(sizeof(CharMap) * size, __FILE__, __LINE__);
    if (!*map)
        return NULL;

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

    // Create output tensor
    int out_sizes[2] = {size, uniqueCount};
    Node *encoded = empty(out_sizes, 2);
    if (!encoded)
    {
        cm_safe_free((void **)map);
        return NULL;
    }

    // Perform one-hot encoding
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < uniqueCount; j++)
        {
            encoded->tensor->storage->data[i * uniqueCount + j] =
                (x[i] == (*map)[j].character) ? 1.0f : 0.0f;
        }
    }

    encoded->requires_grad = 1;
    return encoded;
}

Node *one_hot_decoding_tensor(Node *x, int size, CharMap *map, int mapSize)
{
    if (!x || !map)
    {
        LOG_ERROR("Null pointer argument");
        return NULL;
    }

    int sizes[1] = {size};
    Node *decoded = empty(sizes, 1);
    if (!decoded)
        return NULL;

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < mapSize; j++)
        {
            if (x->tensor->storage->data[i * mapSize + j] > 0.5f)
            {
                decoded->tensor->storage->data[i] = (float)map[j].character;
                break;
            }
        }
    }

    return decoded;
}

void free_one_hot_memory(CharMap *map)
{
    cm_safe_free((void **)&map);
}