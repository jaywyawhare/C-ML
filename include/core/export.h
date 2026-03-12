#ifndef CML_CORE_EXPORT_H
#define CML_CORE_EXPORT_H

#ifdef CML_STATIC_DEFINE
  /* Static library — all symbols visible by default */
  #define CML_API
#else
  #ifdef _WIN32
    #ifdef CML_BUILDING_DLL
      #define CML_API __declspec(dllexport)
    #else
      #define CML_API __declspec(dllimport)
    #endif
  #else
    #ifdef CML_BUILDING_DLL
      #define CML_API __attribute__((visibility("default")))
    #else
      #define CML_API
    #endif
  #endif
#endif

#endif /* CML_CORE_EXPORT_H */
