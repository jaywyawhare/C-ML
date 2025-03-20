#ifndef LABEL_ENCODER_H
#define LABEL_ENCODER_H

typedef struct
{
    char character;
    int encodedValue;
} CharMap;

int *labelEncoder(char *x, int size, CharMap **map, int *mapSize);
char *labelDecoder(int *x, int size, CharMap *map, int mapSize);
void freeLabelMemory(CharMap *map, int *encoded, char *decoded);

#endif 
