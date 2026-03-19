#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "core/onnx.h"

int main(void) {
    printf("ONNX Runtime Tests\n");

#ifdef CML_HAS_ONNX

    /* Test 1: Supported operators return true */
    printf("  test_onnx_op_supported_true...");
    assert(cml_onnx_op_supported("Add") == true);
    assert(cml_onnx_op_supported("Relu") == true);
    assert(cml_onnx_op_supported("MatMul") == true);
    printf(" PASS\n");

    /* Test 2: Unsupported operator returns false */
    printf("  test_onnx_op_supported_false...");
    assert(cml_onnx_op_supported("FakeOp") == false);
    assert(cml_onnx_op_supported("") == false);
    assert(cml_onnx_op_supported("NonExistentOp123") == false);
    printf(" PASS\n");

    /* Test 3: Additional common ops should be supported */
    printf("  test_onnx_common_ops_supported...");
    assert(cml_onnx_op_supported("Sub") == true);
    assert(cml_onnx_op_supported("Mul") == true);
    assert(cml_onnx_op_supported("Div") == true);
    assert(cml_onnx_op_supported("Sigmoid") == true);
    assert(cml_onnx_op_supported("Tanh") == true);
    assert(cml_onnx_op_supported("Softmax") == true);
    assert(cml_onnx_op_supported("Conv") == true);
    assert(cml_onnx_op_supported("Reshape") == true);
    printf(" PASS\n");

    /* Test 4: List supported ops returns a reasonable count */
    printf("  test_onnx_list_supported_ops...");
    const char** ops = NULL;
    int count = 0;
    int ret = cml_onnx_list_supported_ops(&ops, &count);
    assert(ret == 0);
    assert(count > 20);
    assert(ops != NULL);
    printf(" (count=%d) ", count);

    /* Verify that the returned list contains known ops */
    int found_add = 0, found_relu = 0, found_matmul = 0;
    for (int i = 0; i < count; i++) {
        assert(ops[i] != NULL);
        assert(strlen(ops[i]) > 0);
        if (strcmp(ops[i], "Add") == 0) found_add = 1;
        if (strcmp(ops[i], "Relu") == 0) found_relu = 1;
        if (strcmp(ops[i], "MatMul") == 0) found_matmul = 1;
    }
    assert(found_add == 1);
    assert(found_relu == 1);
    assert(found_matmul == 1);
    printf("PASS\n");

    printf("All ONNX runtime tests passed.\n");

#else
    printf("ONNX support not available (CML_HAS_ONNX not defined), skipping tests.\n");
#endif

    return 0;
}
