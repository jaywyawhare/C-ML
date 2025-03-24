#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../../include/Preprocessing/one_hot_encoder.h"
#include "../../include/Core/error_codes.h"

void test_one_hot_encoder()
{
    char input[] = "abac";
    int size = strlen(input);
    int mapSize = 0;
    CharMap *map = NULL;
    int *encoded = one_hot_encoding(input, size, &map, &mapSize);
    assert(mapSize == 3);
    char *decoded = one_hot_decoding(encoded, size, map, mapSize);
    assert(strcmp(decoded, input) == 0);
    free_one_hot_memory(encoded, decoded, map);

    mapSize = 0;
    map = NULL;
    encoded = one_hot_encoding(NULL, 4, &map, &mapSize);
    assert(encoded == (int *)CM_NULL_POINTER_ERROR);

    input[0] = 'a';
    mapSize = 0;
    map = NULL;
    encoded = one_hot_encoding(input, 0, &map, &mapSize);
    assert(encoded == (int *)CM_INVALID_PARAMETER_ERROR);

    printf("one_hot_encoder test passed\n");
}

int main()
{
    printf("Testing one_hot_encoder\n");
    test_one_hot_encoder();
    return CM_SUCCESS;
}
