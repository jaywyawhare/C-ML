#include <stdlib.h>
#include <string.h>

typedef struct {
    char character;
    int encodedValue;
} CharMap;

int* labelEncoder(char *x, int size, CharMap *map, int *mapSize) {
    int uniqueCount = 0;

    for (int i = 0; i < size; i++) {
        int isNew = 1;
        for (int j = 0; j < uniqueCount; j++) {
            if (x[i] == map[j].character) {
                isNew = 0;
                break;
            }
        }
        if (isNew) {
            map[uniqueCount].character = x[i];
            map[uniqueCount].encodedValue = uniqueCount;
            uniqueCount++;
        }
    }

    *mapSize = uniqueCount; 
    
    int *encoded = malloc(sizeof(int) * size);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < *mapSize; j++) {
            if (x[i] == map[j].character) {
                encoded[i] = map[j].encodedValue;
                break;
            }
        }
    }

    return encoded;
}

char* labelDecoder(int *x, int size, CharMap *map, int mapSize) {
    char *decoded = malloc(sizeof(char) * size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < mapSize; j++) {
            if (x[i] == map[j].encodedValue) {
                decoded[i] = map[j].character;
                break;
            }
        }
    }
    return decoded;
}

void freeMemory(CharMap *map, int *encoded, char *decoded) {
    free(map);
    free(encoded);
    free(decoded);
}