#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../../src/Preprocessing/labelEncoder.h"

void test_labelEncoder_normal_case()
{
    char input[] = "abca";
    int size = strlen(input);
    int mapSize = 0;
    CharMap *map = NULL;
    int *encoded = labelEncoder(input, size, &map, &mapSize);
    assert(mapSize == 3);
    char *decoded = labelDecoder(encoded, size, map, mapSize);
    assert(strlen(decoded) == size);
    freeLabelMemory(map, encoded, decoded);
    printf("labelEncoder test passed\n");
}
int main()
{
    printf("Testing labelEncoder\n");
    test_labelEncoder_normal_case();
    return 0;
}
