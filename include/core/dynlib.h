#ifndef CML_CORE_DYNLIB_H
#define CML_CORE_DYNLIB_H

/* Platform shim for runtime library loading. Every backend that dlopen()s an
 * accelerator driver includes this instead of repeating the #ifdef ladder.
 * On unsupported platforms the macros degrade to "nothing loads". */

#ifdef __linux__
#include <dlfcn.h>
#define CML_DLOPEN(path, mode) dlopen(path, mode)
#define CML_DLSYM(handle, symbol) dlsym(handle, symbol)
#define CML_DLCLOSE(handle) dlclose(handle)
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
#elif defined(__APPLE__)
#include <dlfcn.h>
#define CML_DLOPEN(path, mode) dlopen(path, mode)
#define CML_DLSYM(handle, symbol) dlsym(handle, symbol)
#define CML_DLCLOSE(handle) dlclose(handle)
#ifndef RTLD_LAZY
#define RTLD_LAZY 1
#endif
#elif defined(_WIN32)
#include <windows.h>
#define CML_DLOPEN(path, mode) LoadLibraryA(path)
#define CML_DLSYM(handle, symbol) GetProcAddress((HMODULE)handle, symbol)
#define CML_DLCLOSE(handle) FreeLibrary((HMODULE)handle)
#define RTLD_LAZY 0
#else
#define CML_DLOPEN(path, mode) NULL
#define CML_DLSYM(handle, symbol) NULL
#define CML_DLCLOSE(handle) ((void)0)
#define RTLD_LAZY 0
#endif

#endif /* CML_CORE_DYNLIB_H */
