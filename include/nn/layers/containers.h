/**
 * @file containers.h
 * @brief Module containers: ModuleList and ModuleDict
 */

#ifndef CML_NN_LAYERS_CONTAINERS_H
#define CML_NN_LAYERS_CONTAINERS_H

#include "nn.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ModuleList {
    Module base;
    Module** modules;
    int num_modules;
    int capacity;
} ModuleList;

ModuleList* nn_module_list(void);
int module_list_append(ModuleList* list, Module* module);
int module_list_insert(ModuleList* list, int index, Module* module);
Module* module_list_get(ModuleList* list, int index);
int module_list_remove(ModuleList* list, int index);
int module_list_length(ModuleList* list);

typedef struct ModuleDictEntry {
    char* key;
    Module* module;
} ModuleDictEntry;

typedef struct ModuleDict {
    Module base;
    ModuleDictEntry* entries;
    int num_entries;
    int capacity;
} ModuleDict;

ModuleDict* nn_module_dict(void);
int module_dict_add(ModuleDict* dict, const char* key, Module* module);
Module* module_dict_get(ModuleDict* dict, const char* key);
int module_dict_remove(ModuleDict* dict, const char* key);
int module_dict_size(ModuleDict* dict);
const char** module_dict_keys(ModuleDict* dict, int* num_keys);

#ifdef __cplusplus
}
#endif

#endif // CML_NN_LAYERS_CONTAINERS_H
