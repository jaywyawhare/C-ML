#ifndef ONE_HOT_ENCODER_H
#define ONE_HOT_ENCODER_H

#include "../Core/autograd.h"

typedef struct
{
    char character;
    int encodedValue;
} CharMap;

Node *one_hot_encoding_tensor(char *x, int size, CharMap **map, int *mapSize);
Node *one_hot_decoding_tensor(Node *x, int size, CharMap *map, int mapSize);

void free_one_hot_memory(CharMap *map);

#endif
