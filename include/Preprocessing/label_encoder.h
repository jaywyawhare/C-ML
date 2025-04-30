#ifndef LABEL_ENCODER_H
#define LABEL_ENCODER_H

#include "../Core/autograd.h"

typedef struct {
    char character;
    int encodedValue;
} CharMap;

Node* label_encoder_tensor(char* x, int size, CharMap** map, int* mapSize);
Node* label_decoder_tensor(Node* x, int size, CharMap* map, int mapSize);

void free_label_memory(CharMap* map);

#endif
