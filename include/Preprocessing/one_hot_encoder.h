#ifndef ONE_HOT_ENCODER_H
#define ONE_HOT_ENCODER_H

typedef struct
{
    char character;
    int encodedValue;
} CharMap;

int *oneHotEncoding(char *x, int size, CharMap **map, int *mapSize);
char *oneHotDecoding(int *x, int size, CharMap *map, int mapSize);
void freeOneHotMemory(int *x, char *y, CharMap *map);

#endif
