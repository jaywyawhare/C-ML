#ifndef MEMORY_MANAGEMENT_H
#define MEMORY_MANAGEMENT_H

#include<stddef.h>

void *cm_safe_malloc(size_t size, const char *file, int line);

void cm_safe_free(void *ptr);

#endif