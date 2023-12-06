#include <stdbool.h>
#include <string.h>
#include <stdlib.h> 

typedef struct {
    char character;
    int encodedValue;
} CharMap;

int* oneHotEncoding(char *x, int size, CharMap *map, int *mapSize) {
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
    
    int *encoded = malloc(sizeof(int) * size * uniqueCount);
    
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < uniqueCount; j++) {
            if (x[i] == map[j].character) {
                encoded[i * uniqueCount + j] = 1;
            } else {
                encoded[i * uniqueCount + j] = 0;
            }
        }
    }

    return encoded;
}

char* oneHotDecoding(int *x, int size, CharMap *map, int mapSize) {
    char *decoded = malloc(sizeof(char) * (size + 1)); 

    for (int i = 0; i < size; i++) {
        decoded[i] = '\0'; 
        for (int j = 0; j < mapSize; j++) {
            if (x[i * mapSize + j] == 1) {
                decoded[i] = map[j].character;
                break;
            }
        }
    }
    decoded[size] = '\0'; 
    return decoded;
}

void freeMemory(int *x, char *y, CharMap *map) {
    free(x);
    free(y);
    free(map);
}
