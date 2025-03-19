#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef struct
{
    char character;
    int encodedValue;
} CharMap;

int *oneHotEncoding(char *x, int size, CharMap **map, int *mapSize)
{
    if (size == 0)
    {
        fprintf(stderr, "Input size is zero\n");
        return NULL;
    }

    *map = NULL; 
    *map = malloc(sizeof(CharMap) * size);
    if (*map == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
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

    int *encoded = malloc(sizeof(int) * size * uniqueCount);
    if (encoded == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(*map);
        exit(1);
    }

    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < uniqueCount; j++)
        {
            encoded[i * uniqueCount + j] = (x[i] == (*map)[j].character) ? 1 : 0;
        }
    }
    return encoded;
}

char *oneHotDecoding(int *x, int size, CharMap *map, int mapSize)
{
    char *decoded = malloc(sizeof(char) * (size + 1));
    if (decoded == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
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
    return decoded;
}

void freeOneHotMemory(int *x, char *y, CharMap *map)
{
    if (x != NULL)
        free(x);
    if (y != NULL)
        free(y);
    if (map != NULL)
        free(map);
}