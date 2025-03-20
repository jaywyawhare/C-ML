#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../../src/Preprocessing/oneHotEncoder.h"

void test_oneHotEncoder_normal_case()
{
    char input[] = "abac";
    int size = strlen(input);
    int mapSize = 0;
    CharMap *map = NULL;
    int *encoded = oneHotEncoding(input, size, &map, &mapSize);
    assert(mapSize == 3);
    char *decoded = oneHotDecoding(encoded, size, map, mapSize);
    assert(strcmp(decoded, input) == 0);
    freeOneHotMemory(encoded, decoded, map);
    printf("oneHotEncoder test passed\n");
}

int main()
{
    printf("Testing oneHotEncoder\n");
    test_oneHotEncoder_normal_case();
    return 0;
}
