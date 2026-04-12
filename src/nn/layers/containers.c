#include "nn/layers/containers.h"
#include "nn.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static Tensor* module_list_forward(Module* module, Tensor* input) {
    (void)module;
    return input; /* ModuleList doesn't define forward - user iterates */
}

static void module_list_free(Module* module) {
    ModuleList* list = (ModuleList*)module;
    if (!list) return;

    if (list->modules) {
        for (int i = 0; i < list->num_modules; i++) {
            if (list->modules[i]) {
                module_free(list->modules[i]);
            }
        }
        free(list->modules);
    }

    free(list);
}

ModuleList* nn_module_list(void) {
    ModuleList* list = malloc(sizeof(ModuleList));
    if (!list) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for ModuleList", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)list, "ModuleList", module_list_forward, module_list_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ModuleList module", __FILE__,
                         __LINE__, __func__);
        free(list);
        return NULL;
    }

    list->modules     = NULL;
    list->num_modules = 0;
    list->capacity    = 0;
    extern void cml_track_module(Module*);
    cml_track_module((Module*)list);

    return list;
}

int module_list_append(ModuleList* list, Module* module) {
    if (!list || !module) return -1;

    if (list->num_modules >= list->capacity) {
        int new_cap = list->capacity == 0 ? 8 : list->capacity * 2;
        Module** new_mods = realloc(list->modules, (size_t)new_cap * sizeof(Module*));
        if (!new_mods) return -1;
        list->modules  = new_mods;
        list->capacity = new_cap;
    }

    /* Transfer ownership: the list is now responsible for freeing this child. */
    extern void cml_untrack_module(Module*);
    cml_untrack_module(module);

    list->modules[list->num_modules] = module;
    int module_index = list->num_modules;
    list->num_modules++;
    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                char param_name[256];
                snprintf(param_name, sizeof(param_name), "%d.%s.%s", module_index, module->name,
                         params[i]->name ? params[i]->name : "unnamed");
                Tensor* pt = params[i]->tensor;
                nn_tensor_param_alias(pt);
                if (module_add_parameter((Module*)list, pt, param_name,
                                         params[i]->requires_grad) != 0)
                    pt->ref_count--;
            }
        }
        if (params) free(params);
    }

    return 0;
}

int module_list_insert(ModuleList* list, int index, Module* module) {
    if (!list || !module || index < 0 || index > list->num_modules) return -1;

    /* Ensure capacity */
    if (list->num_modules >= list->capacity) {
        int new_cap = list->capacity == 0 ? 8 : list->capacity * 2;
        Module** new_mods = realloc(list->modules, (size_t)new_cap * sizeof(Module*));
        if (!new_mods) return -1;
        list->modules  = new_mods;
        list->capacity = new_cap;
    }

    /* Transfer ownership: the list is now responsible for freeing this child. */
    extern void cml_untrack_module(Module*);
    cml_untrack_module(module);

    /* Shift right */
    for (int i = list->num_modules; i > index; i--) {
        list->modules[i] = list->modules[i - 1];
    }
    list->modules[index] = module;
    list->num_modules++;
    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                char param_name[256];
                snprintf(param_name, sizeof(param_name), "%d.%s.%s", index, module->name,
                         params[i]->name ? params[i]->name : "unnamed");
                Tensor* pt = params[i]->tensor;
                nn_tensor_param_alias(pt);
                if (module_add_parameter((Module*)list, pt, param_name,
                                         params[i]->requires_grad) != 0)
                    pt->ref_count--;
            }
        }
        if (params) free(params);
    }

    return 0;
}

Module* module_list_get(ModuleList* list, int index) {
    if (!list || index < 0 || index >= list->num_modules) return NULL;
    return list->modules[index];
}

int module_list_remove(ModuleList* list, int index) {
    if (!list || index < 0 || index >= list->num_modules) return -1;

    /* Shift left (don't free the module - caller's responsibility) */
    for (int i = index; i < list->num_modules - 1; i++) {
        list->modules[i] = list->modules[i + 1];
    }
    list->num_modules--;
    return 0;
}

int module_list_length(ModuleList* list) {
    return list ? list->num_modules : 0;
}

static Tensor* module_dict_forward(Module* module, Tensor* input) {
    (void)module;
    return input; /* ModuleDict doesn't define forward - user looks up by key */
}

static void module_dict_free(Module* module) {
    ModuleDict* dict = (ModuleDict*)module;
    if (!dict) return;

    if (dict->entries) {
        for (int i = 0; i < dict->num_entries; i++) {
            free(dict->entries[i].key);
            if (dict->entries[i].module) {
                module_free(dict->entries[i].module);
            }
        }
        free(dict->entries);
    }

    free(dict);
}

ModuleDict* nn_module_dict(void) {
    ModuleDict* dict = malloc(sizeof(ModuleDict));
    if (!dict) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR,
                         "Failed to allocate memory for ModuleDict", __FILE__, __LINE__,
                         __func__);
        return NULL;
    }

    if (module_init((Module*)dict, "ModuleDict", module_dict_forward, module_dict_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ModuleDict module", __FILE__,
                         __LINE__, __func__);
        free(dict);
        return NULL;
    }

    dict->entries     = NULL;
    dict->num_entries = 0;
    dict->capacity    = 0;
    extern void cml_track_module(Module*);
    cml_track_module((Module*)dict);

    return dict;
}

int module_dict_add(ModuleDict* dict, const char* key, Module* module) {
    if (!dict || !key || !module) return -1;

    /* Transfer ownership: the dict is now responsible for freeing this child. */
    extern void cml_untrack_module(Module*);
    cml_untrack_module(module);

    for (int i = 0; i < dict->num_entries; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            module_free(dict->entries[i].module);
            dict->entries[i].module = module;
            return 0;
        }
    }
    if (dict->num_entries >= dict->capacity) {
        int new_cap = dict->capacity == 0 ? 8 : dict->capacity * 2;
        ModuleDictEntry* new_entries = realloc(dict->entries,
                                               (size_t)new_cap * sizeof(ModuleDictEntry));
        if (!new_entries) return -1;
        dict->entries  = new_entries;
        dict->capacity = new_cap;
    }

    dict->entries[dict->num_entries].key    = strdup(key);
    dict->entries[dict->num_entries].module = module;
    if (!dict->entries[dict->num_entries].key) return -1;
    dict->num_entries++;
    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) == 0) {
        for (int i = 0; i < num_params; i++) {
            if (params[i]) {
                char param_name[256];
                snprintf(param_name, sizeof(param_name), "%s.%s.%s", key, module->name,
                         params[i]->name ? params[i]->name : "unnamed");
                Tensor* pt = params[i]->tensor;
                nn_tensor_param_alias(pt);
                if (module_add_parameter((Module*)dict, pt, param_name,
                                         params[i]->requires_grad) != 0)
                    pt->ref_count--;
            }
        }
        if (params) free(params);
    }

    return 0;
}

Module* module_dict_get(ModuleDict* dict, const char* key) {
    if (!dict || !key) return NULL;
    for (int i = 0; i < dict->num_entries; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            return dict->entries[i].module;
        }
    }
    return NULL;
}

int module_dict_remove(ModuleDict* dict, const char* key) {
    if (!dict || !key) return -1;

    for (int i = 0; i < dict->num_entries; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            free(dict->entries[i].key);
            /* Don't free module - caller's responsibility */
            for (int j = i; j < dict->num_entries - 1; j++) {
                dict->entries[j] = dict->entries[j + 1];
            }
            dict->num_entries--;
            return 0;
        }
    }
    return -1;
}

int module_dict_size(ModuleDict* dict) {
    return dict ? dict->num_entries : 0;
}

const char** module_dict_keys(ModuleDict* dict, int* num_keys) {
    if (!dict || !num_keys) return NULL;
    *num_keys = dict->num_entries;
    if (dict->num_entries == 0) return NULL;

    const char** keys = malloc((size_t)dict->num_entries * sizeof(const char*));
    if (!keys) return NULL;

    for (int i = 0; i < dict->num_entries; i++) {
        keys[i] = dict->entries[i].key;
    }
    return keys;
}
