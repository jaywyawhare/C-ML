#include "nn/layers/containers.h"
#include "nn.h"
#include "core/logging.h"
#include "core/error_stack.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "alloc/cml_allocator.h"

/** torch.nn.ModuleList has no forward; this returns the input unchanged (the user iterates). */
static Tensor* module_list_forward(Module* module, Tensor* input) {
    (void)module;
    return input; /* ModuleList doesn't define forward - user iterates */
}

/** Free a ModuleList and every child module it owns. */
static void module_list_free(Module* module) {
    ModuleList* list = (ModuleList*)module;
    if (!list)
        return;

    if (list->modules) {
        for (int i = 0; i < list->num_modules; i++) {
            if (list->modules[i]) {
                module_free(list->modules[i]);
            }
        }
        cml_free(list->modules);
    }

    cml_free(list);
}

/** Construct an empty ModuleList (tracked for global cleanup). NULL on failure. */
ModuleList* nn_module_list(void) {
    ModuleList* list = cml_malloc(sizeof(ModuleList));
    if (!list) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for ModuleList",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)list, "ModuleList", module_list_forward, module_list_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ModuleList module", __FILE__,
                         __LINE__, __func__);
        cml_free(list);
        return NULL;
    }

    list->modules     = NULL;
    list->num_modules = 0;
    list->capacity    = 0;
    extern void cml_track_module(Module*);
    cml_track_module((Module*)list);

    return list;
}

/** Re-export `module`'s parameters on `list` under "<index>.<module>.<param>"
 *  names, aliasing rather than copying the tensors. */
static void module_list_adopt_params(ModuleList* list, Module* module, int index) {
    Parameter** params = NULL;
    int num_params     = 0;
    if (module_collect_parameters(module, &params, &num_params, true) != 0)
        return;

    for (int i = 0; i < num_params; i++) {
        if (!params[i])
            continue;
        char param_name[256];
        snprintf(param_name, sizeof(param_name), "%d.%s.%s", index, module->name,
                 params[i]->name ? params[i]->name : "unnamed");
        Tensor* pt = params[i]->tensor;
        nn_tensor_param_alias(pt);
        if (module_add_parameter((Module*)list, pt, param_name, params[i]->requires_grad) != 0)
            pt->ref_count--;
    }

    if (params)
        cml_free(params);
}

/** Append a module, taking ownership and adopting its parameters. Returns -1 on NULL args or
 *  allocation failure. */
int module_list_append(ModuleList* list, Module* module) {
    if (!list || !module)
        return -1;

    if (list->num_modules >= list->capacity) {
        int new_cap       = list->capacity == 0 ? 8 : list->capacity * 2;
        Module** new_mods = cml_realloc(list->modules, (size_t)new_cap * sizeof(Module*));
        if (!new_mods)
            return -1;
        list->modules  = new_mods;
        list->capacity = new_cap;
    }

    /* Transfer ownership: the list is now responsible for freeing this child. */
    extern void cml_untrack_module(Module*);
    cml_untrack_module(module);

    list->modules[list->num_modules] = module;
    int module_index                 = list->num_modules;
    list->num_modules++;
    module_list_adopt_params(list, module, module_index);

    return 0;
}

/** Insert a module at index, taking ownership and adopting its parameters. Returns -1 on NULL
 *  args, out-of-range index, or allocation failure. */
int module_list_insert(ModuleList* list, int index, Module* module) {
    if (!list || !module || index < 0 || index > list->num_modules)
        return -1;

    /* Ensure capacity */
    if (list->num_modules >= list->capacity) {
        int new_cap       = list->capacity == 0 ? 8 : list->capacity * 2;
        Module** new_mods = cml_realloc(list->modules, (size_t)new_cap * sizeof(Module*));
        if (!new_mods)
            return -1;
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
    module_list_adopt_params(list, module, index);

    return 0;
}

/** Module at index, or NULL if the list is NULL or the index is out of range. */
Module* module_list_get(ModuleList* list, int index) {
    if (!list || index < 0 || index >= list->num_modules)
        return NULL;
    return list->modules[index];
}

/** Remove the module at index without freeing it (caller takes ownership). Returns -1 on NULL
 *  list or out-of-range index. */
int module_list_remove(ModuleList* list, int index) {
    if (!list || index < 0 || index >= list->num_modules)
        return -1;

    /* Shift left (don't free the module - caller's responsibility) */
    for (int i = index; i < list->num_modules - 1; i++) {
        list->modules[i] = list->modules[i + 1];
    }
    list->num_modules--;
    return 0;
}

/** Number of modules in the list, or 0 if NULL. */
int module_list_length(ModuleList* list) { return list ? list->num_modules : 0; }

/** torch.nn.ModuleDict has no forward; this returns the input unchanged (user looks up by key). */
static Tensor* module_dict_forward(Module* module, Tensor* input) {
    (void)module;
    return input; /* ModuleDict doesn't define forward - user looks up by key */
}

/** Free a ModuleDict, its key strings, and every child module it owns. */
static void module_dict_free(Module* module) {
    ModuleDict* dict = (ModuleDict*)module;
    if (!dict)
        return;

    if (dict->entries) {
        for (int i = 0; i < dict->num_entries; i++) {
            cml_free(dict->entries[i].key);
            if (dict->entries[i].module) {
                module_free(dict->entries[i].module);
            }
        }
        cml_free(dict->entries);
    }

    cml_free(dict);
}

/** Construct an empty ModuleDict (tracked for global cleanup). NULL on failure. */
ModuleDict* nn_module_dict(void) {
    ModuleDict* dict = cml_malloc(sizeof(ModuleDict));
    if (!dict) {
        error_stack_push(CM_MEMORY_ALLOCATION_ERROR, "Failed to allocate memory for ModuleDict",
                         __FILE__, __LINE__, __func__);
        return NULL;
    }

    if (module_init((Module*)dict, "ModuleDict", module_dict_forward, module_dict_free) != 0) {
        error_stack_push(CM_OPERATION_FAILED, "Failed to initialize ModuleDict module", __FILE__,
                         __LINE__, __func__);
        cml_free(dict);
        return NULL;
    }

    dict->entries     = NULL;
    dict->num_entries = 0;
    dict->capacity    = 0;
    extern void cml_track_module(Module*);
    cml_track_module((Module*)dict);

    return dict;
}

/** Add a module under key, taking ownership and adopting its parameters; an existing key is
 *  replaced (old module freed). Returns -1 on NULL args or allocation failure. */
int module_dict_add(ModuleDict* dict, const char* key, Module* module) {
    if (!dict || !key || !module)
        return -1;

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
        ModuleDictEntry* new_entries =
            cml_realloc(dict->entries, (size_t)new_cap * sizeof(ModuleDictEntry));
        if (!new_entries)
            return -1;
        dict->entries  = new_entries;
        dict->capacity = new_cap;
    }

    dict->entries[dict->num_entries].key    = cml_strdup(key);
    dict->entries[dict->num_entries].module = module;
    if (!dict->entries[dict->num_entries].key)
        return -1;
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
                if (module_add_parameter((Module*)dict, pt, param_name, params[i]->requires_grad) !=
                    0)
                    pt->ref_count--;
            }
        }
        if (params)
            cml_free(params);
    }

    return 0;
}

/** Module stored under key, or NULL if absent or on NULL args. */
Module* module_dict_get(ModuleDict* dict, const char* key) {
    if (!dict || !key)
        return NULL;
    for (int i = 0; i < dict->num_entries; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            return dict->entries[i].module;
        }
    }
    return NULL;
}

/** Remove the entry for key, freeing the key string but not the module (caller takes ownership).
 *  Returns -1 on NULL args or if the key is absent. */
int module_dict_remove(ModuleDict* dict, const char* key) {
    if (!dict || !key)
        return -1;

    for (int i = 0; i < dict->num_entries; i++) {
        if (strcmp(dict->entries[i].key, key) == 0) {
            cml_free(dict->entries[i].key);
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

/** Number of entries in the dict, or 0 if NULL. */
int module_dict_size(ModuleDict* dict) { return dict ? dict->num_entries : 0; }

/** Allocate and return an array of the dict's keys (borrowed pointers; caller frees the array),
 *  writing the count to num_keys. NULL on NULL args, empty dict, or allocation failure. */
const char** module_dict_keys(ModuleDict* dict, int* num_keys) {
    if (!dict || !num_keys)
        return NULL;
    *num_keys = dict->num_entries;
    if (dict->num_entries == 0)
        return NULL;

    const char** keys = cml_malloc((size_t)dict->num_entries * sizeof(const char*));
    if (!keys)
        return NULL;

    for (int i = 0; i < dict->num_entries; i++) {
        keys[i] = dict->entries[i].key;
    }
    return keys;
}
