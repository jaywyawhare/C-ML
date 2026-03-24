#include "tensor/dtype_vec.h"
#include <stdio.h>
#include <string.h>

static bool is_valid_width(int n) {
    return n == 1 || n == 2 || n == 4 || n == 8 || n == 16;
}

VecDType dtype_vec(DType d, int n) {
    VecDType vt;
    vt.scalar = d;
    vt.n = is_valid_width(n) ? n : 0;
    return vt;
}

bool dtype_vec_valid(VecDType vt) {
    if (!is_valid_width(vt.n)) return false;
    
    if (vt.scalar >= DTYPE_FLOAT8_E4M3 && vt.n > 2) return false;
    
    if (vt.scalar == DTYPE_BOOL && vt.n > 1) return false;
    return true;
}

size_t dtype_vec_size(VecDType vt) {
    return cml_dtype_size(vt.scalar) * (size_t)vt.n;
}

size_t dtype_vec_alignment(VecDType vt) {
    size_t sz = dtype_vec_size(vt);
    
    size_t align = 1;
    while (align < sz && align < 16) align <<= 1;
    return align;
}

static const char* dtype_c_base(DType d) {
    switch (d) {
        case DTYPE_FLOAT32:       return "float";
        case DTYPE_FLOAT64:       return "double";
        case DTYPE_INT32:         return "int";
        case DTYPE_INT64:         return "long";
        case DTYPE_BOOL:          return "bool";
        case DTYPE_FLOAT16:       return "half";
        case DTYPE_BFLOAT16:      return "__bf16";
        case DTYPE_INT8:          return "char";
        case DTYPE_UINT8:         return "uchar";
        case DTYPE_INT16:         return "short";
        case DTYPE_UINT16:        return "ushort";
        case DTYPE_UINT32:        return "uint";
        case DTYPE_UINT64:        return "ulong";
        case DTYPE_FLOAT8_E4M3:   return "__fp8_e4m3";
        case DTYPE_FLOAT8_E5M2:   return "__fp8_e5m2";
        case DTYPE_FLOAT8_E4M3_FNUZ: return "__fp8_e4m3_fnuz";
        case DTYPE_FLOAT8_E5M2_FNUZ: return "__fp8_e5m2_fnuz";
        default:                  return "unknown";
    }
}

const char* dtype_vec_c_name(VecDType vt) {
    if (!dtype_vec_valid(vt)) return "";
    if (vt.n == 1) return dtype_c_base(vt.scalar);

    
    static const struct { DType d; int n; const char* name; } table[] = {
        
        { DTYPE_FLOAT32, 2,  "float2"  },
        { DTYPE_FLOAT32, 4,  "float4"  },
        { DTYPE_FLOAT32, 8,  "float8"  },
        { DTYPE_FLOAT32, 16, "float16" },
        
        { DTYPE_FLOAT16, 2,  "half2"   },
        { DTYPE_FLOAT16, 4,  "half4"   },
        { DTYPE_FLOAT16, 8,  "half8"   },
        { DTYPE_FLOAT16, 16, "half16"  },
        
        { DTYPE_BFLOAT16, 2, "__bf16x2"  },
        { DTYPE_BFLOAT16, 4, "__bf16x4"  },
        
        { DTYPE_INT32, 2,  "int2"   },
        { DTYPE_INT32, 4,  "int4"   },
        { DTYPE_INT32, 8,  "int8"   },
        { DTYPE_INT32, 16, "int16"  },
        
        { DTYPE_UINT32, 2, "uint2"  },
        { DTYPE_UINT32, 4, "uint4"  },
        { DTYPE_UINT32, 8, "uint8"  },
        
        { DTYPE_INT8, 2,  "char2"  },
        { DTYPE_INT8, 4,  "char4"  },
        { DTYPE_INT8, 8,  "char8"  },
        { DTYPE_INT8, 16, "char16" },
        
        { DTYPE_UINT8, 2,  "uchar2"  },
        { DTYPE_UINT8, 4,  "uchar4"  },
        { DTYPE_UINT8, 8,  "uchar8"  },
        { DTYPE_UINT8, 16, "uchar16" },
        
        { DTYPE_INT16, 2,  "short2"  },
        { DTYPE_INT16, 4,  "short4"  },
        { DTYPE_INT16, 8,  "short8"  },
        
        { DTYPE_UINT16, 2, "ushort2" },
        { DTYPE_UINT16, 4, "ushort4" },
        { DTYPE_UINT16, 8, "ushort8" },
        
        { DTYPE_INT64, 2, "long2"   },
        { DTYPE_INT64, 4, "long4"   },
        
        { DTYPE_FLOAT64, 2, "double2" },
        { DTYPE_FLOAT64, 4, "double4" },
        
        { DTYPE_FLOAT8_E4M3, 2, "__fp8_e4m3x2" },
        { DTYPE_FLOAT8_E5M2, 2, "__fp8_e5m2x2" },
    };
    for (size_t i = 0; i < sizeof(table)/sizeof(table[0]); ++i)
        if (table[i].d == vt.scalar && table[i].n == vt.n)
            return table[i].name;
    return "";
}

VecDType dtype_vec_widen(VecDType vt) {
    int new_n = vt.n * 2;
    if (!is_valid_width(new_n)) new_n = vt.n;
    return dtype_vec(vt.scalar, new_n);
}

VecDType dtype_vec_narrow(VecDType vt) {
    int new_n = vt.n / 2;
    if (new_n < 1) new_n = 1;
    return dtype_vec(vt.scalar, new_n);
}

int dtype_vec_max_width(DType d) {
    switch (d) {
        case DTYPE_FLOAT32: return 16;
        case DTYPE_FLOAT16: return 16;
        case DTYPE_INT32:   return 16;
        case DTYPE_INT8:    return 16;
        case DTYPE_UINT8:   return 16;
        case DTYPE_BFLOAT16: return 4;
        case DTYPE_FLOAT8_E4M3:
        case DTYPE_FLOAT8_E5M2:
        case DTYPE_FLOAT8_E4M3_FNUZ:
        case DTYPE_FLOAT8_E5M2_FNUZ: return 2;
        case DTYPE_BOOL: return 1;
        default: return 4;
    }
}

int dtype_vec_splat(VecDType vt, double val, char* buf, size_t buf_size) {
    if (!buf || buf_size == 0) return -1;
    const char* tname = dtype_vec_c_name(vt);
    if (!tname || !tname[0]) return -1;
    if (vt.n == 1)
        return snprintf(buf, buf_size, "(%s)(%g)", tname, val) < 0 ? -1 : 0;
    return snprintf(buf, buf_size, "(%s)(%g)", tname, val) < 0 ? -1 : 0;
}

int dtype_vec_lane(VecDType vt, int lane, const char* vec_name,
                   char* buf, size_t buf_size) {
    if (!buf || buf_size == 0 || !vec_name) return -1;
    if (lane < 0 || lane >= vt.n) return -1;
    
    if (vt.n <= 4) {
        const char* lanes[] = {"x", "y", "z", "w"};
        return snprintf(buf, buf_size, "%s.%s", vec_name, lanes[lane]) < 0 ? -1 : 0;
    }
    return snprintf(buf, buf_size, "%s.s%x", vec_name, lane) < 0 ? -1 : 0;
}
