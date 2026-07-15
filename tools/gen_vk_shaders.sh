#!/usr/bin/env bash
# Regenerate include/ops/ir/gpu/vk_shaders.h from the GLSL compute shaders.
set -e
cd "$(dirname "$0")/.."
D=src/ops/ir/gpu/shaders
for s in binary unary matmul; do glslc -O -fshader-stage=comp "$D/$s.comp" -o "/tmp/$s.spv"; done
python3 - <<'PY'
import struct
def emit(f,name,path):
    d=open(path,'rb').read(); w=[struct.unpack('<I',d[i:i+4])[0] for i in range(0,len(d),4)]
    f.write('static const uint32_t %s[] = {\n'%name)
    for i in range(0,len(w),8): f.write('  '+','.join('0x%08xu'%x for x in w[i:i+8])+',\n')
    f.write('};\nstatic const unsigned %s_SIZE = %d;\n\n'%(name,len(d)))
with open('include/ops/ir/gpu/vk_shaders.h','w') as f:
    f.write('/* Auto-generated from src/ops/ir/gpu/shaders/*.comp via glslc. */\n#ifndef CML_VK_SHADERS_H\n#define CML_VK_SHADERS_H\n#include <stdint.h>\n\n')
    emit(f,'VK_SPV_BINARY','/tmp/binary.spv'); emit(f,'VK_SPV_UNARY','/tmp/unary.spv'); emit(f,'VK_SPV_MATMUL','/tmp/matmul.spv')
    f.write('#endif\n')
PY
echo "regenerated vk_shaders.h"
