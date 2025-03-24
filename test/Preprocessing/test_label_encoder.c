#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "../../include/Preprocessing/label_encoder.h"
#include "../../include/Core/error_codes.h"

void test_label_encoder()
{
    char input[] = "abca";
    int size = strlen(input);
    int mapSize = 0;
    CharMap *map = NULL;
    int *encoded = label_encoder(input, size, &map, &mapSize);
    assert(mapSize == 3);
    char *decoded = label_decoder(encoded, size, map, mapSize);
    assert(strlen(decoded) == size);
    free_label_memory(map, encoded, decoded);

    mapSize = 0;
    map = NULL;
    encoded = label_encoder(NULL, 4, &map, &mapSize);
    assert(encoded == (int *)CM_NULL_POINTER_ERROR);

    input[0] = 'a';
    mapSize = 0;
    map = NULL;
    encoded = label_encoder(input, 0, &map, &mapSize);
    assert(encoded == (int *)CM_INVALID_PARAMETER_ERROR);

    printf("label_encoder test passed\n");
}

int main()
{
    printf("Testing label_encoder\n");
    test_label_encoder();
    return 0;
}
