#include <stdlib.h>
#include <string.h>
#include "../../include/Preprocessing/label_encoder.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

Node *label_encoder_tensor(char *x, int size, CharMap **map, int *mapSize)
{
    if (!x || !map || !mapSize)
    {
        LOG_ERROR("Null pointer argument");
        return NULL;
    }

    // Create map of unique characters
    *map = (CharMap *)cm_safe_malloc(sizeof(CharMap) * size, __FILE__, __LINE__);
    if (!*map)
        return NULL;

    // Build character map
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
    int sizes[1] = {size};
    Node *output = empty(sizes, 1);
    if (!output)
    {
        cm_safe_free((void **)map);
        return NULL;
    }

    // Encode using tensor operations
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < *mapSize; j++)
        {
            if (x[i] == (*map)[j].character)
            {
                output->tensor->storage->data[i] = (float)(*map)[j].encodedValue;
                break;
            }
        }
    }

    output->requires_grad = 1;
    return output;
}

Node *label_decoder_tensor(Node *x, int size, CharMap *map, int mapSize)
{
    if (!x || !map)
    {
        LOG_ERROR("Null pointer argument");
        return NULL;
    }

    int sizes[1] = {size};
    Node *output = empty(sizes, 1);
    if (!output)
        return NULL;

    for (int i = 0; i < size; i++)
    {
        int encoded_val = (int)x->tensor->storage->data[i];
        for (int j = 0; j < mapSize; j++)
        {
            if (encoded_val == map[j].encodedValue)
            {
                output->tensor->storage->data[i] = (float)map[j].character;
                break;
            }
        }
    }

    return output;
}

void free_label_memory(CharMap *map)
{
    cm_safe_free((void **)&map);
}