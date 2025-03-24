#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"

void test_cm_safe_malloc()
{
    int *ptr = (int *)cm_safe_malloc(sizeof(int), __FILE__, __LINE__);
    assert(ptr != (void *)CM_MEMORY_ALLOCATION_ERROR);

    *ptr = 42;
    assert(*ptr == 42);

    cm_safe_free((void **)&ptr);

    assert(ptr == NULL);

    void *zero_ptr = cm_safe_malloc(0, __FILE__, __LINE__);
    assert(zero_ptr != (void *)CM_MEMORY_ALLOCATION_ERROR);
    cm_safe_free(&zero_ptr);

    size_t large_size = (size_t)-1;
    void *large_ptr = cm_safe_malloc(large_size, __FILE__, __LINE__);
    assert(large_ptr == (void *)CM_MEMORY_ALLOCATION_ERROR);

    printf("cm_safe_malloc test passed\n");
}

void test_cm_safe_free()
{
    cm_safe_free(NULL);

    int *ptr = (int *)cm_safe_malloc(sizeof(int), __FILE__, __LINE__);
    assert(ptr != (void *)CM_MEMORY_ALLOCATION_ERROR);
    *ptr = 99;
    cm_safe_free((void **)&ptr);
    assert(ptr == NULL);

    cm_safe_free(NULL);

    ptr = (int *)cm_safe_malloc(sizeof(int), __FILE__, __LINE__);
    assert(ptr != (void *)CM_MEMORY_ALLOCATION_ERROR);
    cm_safe_free((void **)&ptr);
    cm_safe_free((void **)&ptr);

    printf("cm_safe_free test passed\n");
}

int main()
{
    printf("Testing Memory Management\n");
    test_cm_safe_malloc();
    test_cm_safe_free();
    return 0;
}
