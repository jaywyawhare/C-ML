#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int *labelEncoder(char *x, int size, CharMap **map, int *mapSize)
{
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

    int *encoded = malloc(sizeof(int) * size);
    if (encoded == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        free(*map);
        exit(1);
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
    return encoded;
}

char *labelDecoder(int *x, int size, CharMap *map, int mapSize)
{
    char *decoded = malloc(sizeof(char) * (size + 1));
    if (decoded == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
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
    return decoded;
}

void freeLabelMemory(CharMap *map, int *encoded, char *decoded)
{
    free(map);
    free(encoded);
    free(decoded);
}