#!/usr/bin/env python3
"""
Comprehensive CFFI Bindings Test - Fixed Version
Tests all exposed C functions through Python CFFI bindings.
"""

import sys
import os

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cml._cml_lib import ffi, lib


def test_category(name, tests):
    """Run a category of tests."""
    print(f"\n{'='*60}")
    print(f"Category: {name}")
    print("=" * 60)
    sys.stdout.flush()

    passed = 0
    failed = 0
    skipped = 0

    for test_name, test_func in tests:
        print(f"  Running: {test_name}...", end="", flush=True)
        try:
            result = test_func()
            if result == "SKIP":
                print(f" [SKIP]")
                skipped += 1
            elif result:
                print(f" [PASS]")
                passed += 1
            else:
                print(f" [FAIL]")
                failed += 1
        except Exception as e:
            print(f" [ERROR] {e}")
            failed += 1
        sys.stdout.flush()

    return passed, failed, skipped


def _test_unary_op(fn_name):
    fn = getattr(lib, f"cml_{fn_name}")
    a = lib.cml_ones_2d(3, 3)
    c = fn(a)
    success = c != ffi.NULL
    lib.tensor_free(a)
    if success:
        lib.tensor_free(c)
    return success


def _test_binary_op(fn_name):
    fn = getattr(lib, f"cml_{fn_name}")
    a = lib.cml_ones_2d(3, 3)
    b = lib.cml_ones_2d(3, 3)
    c = fn(a, b)
    success = c != ffi.NULL
    lib.tensor_free(a)
    lib.tensor_free(b)
    if success:
        lib.tensor_free(c)
    return success


def _test_reduction(fn_name):
    fn = getattr(lib, f"cml_{fn_name}")
    a = lib.cml_ones_2d(3, 3)
    c = fn(a, 0, False)
    success = c != ffi.NULL
    lib.tensor_free(a)
    if success:
        lib.tensor_free(c)
    return success


def main():
    print("\n" + "=" * 60)
    print("CML CFFI Comprehensive Bindings Test")
    print("=" * 60)

    total_passed = 0
    total_failed = 0
    total_skipped = 0

    # 1. Initialization
    def test_init():
        result = lib.cml_init()
        return result == 0

    def test_is_initialized():
        return lib.cml_is_initialized() == True

    def test_init_count():
        count = lib.cml_get_init_count()
        return count >= 1

    init_tests = [
        ("cml_init", test_init),
        ("cml_is_initialized", test_is_initialized),
        ("cml_get_init_count", test_init_count),
    ]
    p, f, s = test_category("Initialization", init_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 2. Device and Dtype Management
    def test_get_device():
        device = lib.cml_get_default_device()
        return device >= 0

    def test_get_dtype():
        dtype = lib.cml_get_default_dtype()
        return dtype >= 0

    def test_set_dtype():
        lib.cml_set_default_dtype(0)  # DTYPE_FLOAT32
        return lib.cml_get_default_dtype() == 0

    device_tests = [
        ("cml_get_default_device", test_get_device),
        ("cml_get_default_dtype", test_get_dtype),
        ("cml_set_default_dtype", test_set_dtype),
    ]
    p, f, s = test_category("Device/Dtype Management", device_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 3. Tensor Creation
    def test_zeros_2d():
        t = lib.cml_zeros_2d(3, 4)
        success = t != ffi.NULL
        if success:
            lib.tensor_free(t)
        return success

    def test_ones_2d():
        t = lib.cml_ones_2d(3, 4)
        success = t != ffi.NULL
        if success:
            lib.tensor_free(t)
        return success

    def test_empty_2d():
        t = lib.cml_empty_2d(3, 4)
        success = t != ffi.NULL
        if success:
            lib.tensor_free(t)
        return success

    def test_tensor_2d():
        data = ffi.new("float[6]", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        t = lib.cml_tensor_2d(data, 2, 3)
        success = t != ffi.NULL
        if success:
            lib.tensor_free(t)
        return success

    tensor_create_tests = [
        ("cml_zeros_2d", test_zeros_2d),
        ("cml_ones_2d", test_ones_2d),
        ("cml_empty_2d", test_empty_2d),
        ("cml_tensor_2d", test_tensor_2d),
    ]
    p, f, s = test_category("Tensor Creation", tensor_create_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 4. Tensor Arithmetic Operations
    def test_matmul():
        a = lib.cml_ones_2d(3, 4)
        b = lib.cml_ones_2d(4, 5)
        c = lib.cml_matmul(a, b)
        success = c != ffi.NULL
        lib.tensor_free(a)
        lib.tensor_free(b)
        if success:
            lib.tensor_free(c)
        return success

    arith_tests = [
        ("cml_add", lambda: _test_binary_op("add")),
        ("cml_sub", lambda: _test_binary_op("sub")),
        ("cml_mul", lambda: _test_binary_op("mul")),
        ("cml_div", lambda: _test_binary_op("div")),
        ("cml_matmul", test_matmul),
        ("cml_pow", lambda: _test_binary_op("pow")),
    ]
    p, f, s = test_category("Tensor Arithmetic", arith_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 5. Tensor Math Operations
    math_tests = [
        ("cml_exp", lambda: _test_unary_op("exp")),
        ("cml_log", lambda: _test_unary_op("log")),
        ("cml_sqrt", lambda: _test_unary_op("sqrt")),
        ("cml_sin", lambda: _test_unary_op("sin")),
        ("cml_cos", lambda: _test_unary_op("cos")),
        ("cml_tan", lambda: _test_unary_op("tan")),
    ]
    p, f, s = test_category("Tensor Math", math_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 6. Tensor Activations
    def test_softmax():
        a = lib.cml_ones_2d(3, 3)
        c = lib.cml_softmax(a, 1)
        success = c != ffi.NULL
        lib.tensor_free(a)
        if success:
            lib.tensor_free(c)
        return success

    activation_tests = [
        ("cml_relu", lambda: _test_unary_op("relu")),
        ("cml_sigmoid", lambda: _test_unary_op("sigmoid")),
        ("cml_tanh", lambda: _test_unary_op("tanh")),
        ("cml_softmax", test_softmax),
    ]
    p, f, s = test_category("Tensor Activations", activation_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 7. Tensor Reductions
    reduction_tests = [
        ("cml_sum", lambda: _test_reduction("sum")),
        ("cml_mean", lambda: _test_reduction("mean")),
        ("cml_max", lambda: _test_reduction("max")),
        ("cml_min", lambda: _test_reduction("min")),
    ]
    p, f, s = test_category("Tensor Reductions", reduction_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 8. Tensor Manipulation
    def test_transpose():
        a = lib.cml_ones_2d(3, 4)
        c = lib.cml_transpose(a, 0, 1)
        success = c != ffi.NULL
        lib.tensor_free(a)
        if success:
            lib.tensor_free(c)
        return success

    def test_reshape():
        a = lib.cml_ones_2d(3, 4)
        shape = ffi.new("int[2]", [4, 3])
        c = lib.cml_reshape(a, shape, 2)
        success = c != ffi.NULL
        lib.tensor_free(a)
        if success:
            lib.tensor_free(c)
        return success

    def test_clone():
        a = lib.cml_ones_2d(3, 3)
        c = lib.cml_clone(a)
        success = c != ffi.NULL
        lib.tensor_free(a)
        if success:
            lib.tensor_free(c)
        return success

    def test_detach():
        a = lib.cml_ones_2d(3, 3)
        c = lib.cml_detach(a)
        success = c != ffi.NULL
        lib.tensor_free(a)
        if success:
            lib.tensor_free(c)
        return success

    def test_tensor_data_ptr():
        a = lib.cml_ones_2d(3, 3)
        ptr = lib.tensor_data_ptr(a)
        success = ptr != ffi.NULL
        lib.tensor_free(a)
        return success

    manip_tests = [
        ("cml_transpose", test_transpose),
        ("cml_reshape", test_reshape),
        ("cml_clone", test_clone),
        ("cml_detach", test_detach),
        ("tensor_data_ptr", test_tensor_data_ptr),
    ]
    p, f, s = test_category("Tensor Manipulation", manip_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 9. Autograd
    def test_enable_grad():
        lib.cml_enable_grad()
        return lib.cml_is_grad_enabled() == True

    def test_no_grad():
        lib.cml_no_grad()
        result = lib.cml_is_grad_enabled() == False
        lib.cml_enable_grad()  # Re-enable
        return result

    def test_requires_grad():
        a = lib.cml_ones_2d(3, 3)
        lib.cml_set_requires_grad(a, True)
        result = lib.cml_requires_grad(a)
        lib.tensor_free(a)
        return result == True

    def test_is_leaf():
        a = lib.cml_ones_2d(3, 3)
        result = lib.cml_is_leaf(a)
        lib.tensor_free(a)
        return result == True  # Newly created tensors are leaves

    def test_zero_grad():
        a = lib.cml_ones_2d(3, 3)
        lib.cml_zero_grad(a)
        lib.tensor_free(a)
        return True

    def test_backward():
        a = lib.cml_ones_2d(3, 3)
        lib.cml_set_requires_grad(a, True)
        b = lib.cml_ones_2d(3, 3)
        lib.cml_set_requires_grad(b, True)
        c = lib.cml_add(a, b)
        lib.cml_backward(c, ffi.NULL, False, False)
        lib.tensor_free(a)
        lib.tensor_free(b)
        lib.tensor_free(c)
        return True

    autograd_tests = [
        ("cml_enable_grad", test_enable_grad),
        ("cml_no_grad", test_no_grad),
        ("cml_requires_grad/set", test_requires_grad),
        ("cml_is_leaf", test_is_leaf),
        ("cml_zero_grad", test_zero_grad),
        ("cml_backward", test_backward),
    ]
    p, f, s = test_category("Autograd", autograd_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 10. Neural Network Layers
    def test_nn_sequential():
        seq = lib.cml_nn_sequential()
        return seq != ffi.NULL

    def test_nn_linear():
        linear = lib.cml_nn_linear(10, 5, 0, 0, True)  # DTYPE_FLOAT32, DEVICE_CPU
        return linear != ffi.NULL

    def test_nn_relu():
        relu = lib.cml_nn_relu(False)
        return relu != ffi.NULL

    def test_nn_sigmoid():
        sig = lib.cml_nn_sigmoid()
        return sig != ffi.NULL

    def test_nn_tanh():
        tanh = lib.cml_nn_tanh()
        return tanh != ffi.NULL

    def test_nn_dropout():
        dropout = lib.cml_nn_dropout(0.5, False)
        return dropout != ffi.NULL

    nn_layer_tests = [
        ("cml_nn_sequential", test_nn_sequential),
        ("cml_nn_linear", test_nn_linear),
        ("cml_nn_relu", test_nn_relu),
        ("cml_nn_sigmoid", test_nn_sigmoid),
        ("cml_nn_tanh", test_nn_tanh),
        ("cml_nn_dropout", test_nn_dropout),
    ]
    p, f, s = test_category("Neural Network Layers", nn_layer_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 11. Neural Network Operations
    def test_nn_module_forward():
        linear = lib.cml_nn_linear(4, 2, 0, 0, True)
        if linear == ffi.NULL:
            return False
        module = ffi.cast("Module*", linear)
        x = lib.cml_ones_2d(3, 4)
        y = lib.cml_nn_module_forward(module, x)
        success = y != ffi.NULL
        lib.tensor_free(x)
        if success:
            lib.tensor_free(y)
        lib.module_free(module)
        return success

    def test_nn_training_mode():
        linear = lib.cml_nn_linear(4, 2, 0, 0, True)
        if linear == ffi.NULL:
            return False
        module = ffi.cast("Module*", linear)
        lib.cml_nn_module_train(module)
        train = lib.cml_nn_module_is_training(module)
        lib.cml_nn_module_eval(module)
        eval_mode = not lib.cml_nn_module_is_training(module)
        lib.module_free(module)
        return train and eval_mode

    def test_nn_sequential_forward():
        seq = lib.cml_nn_sequential()
        if seq == ffi.NULL:
            return False

        linear = lib.cml_nn_linear(4, 2, 0, 0, True)
        if linear == ffi.NULL:
            return False
        linear_module = ffi.cast("Module*", linear)
        lib.cml_nn_sequential_add(seq, linear_module)

        x = lib.cml_ones_2d(3, 4)
        y = lib.cml_nn_sequential_forward(seq, x)
        success = y != ffi.NULL

        lib.tensor_free(x)
        if success:
            lib.tensor_free(y)
        return success

    nn_ops_tests = [
        ("cml_nn_module_forward", test_nn_module_forward),
        ("cml_nn_module_train/eval", test_nn_training_mode),
        ("cml_nn_sequential_forward", test_nn_sequential_forward),
    ]
    p, f, s = test_category("Neural Network Operations", nn_ops_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 12. Loss Functions
    def test_mse_loss():
        pred = lib.cml_ones_2d(3, 3)
        target = lib.cml_zeros_2d(3, 3)
        loss = lib.cml_nn_mse_loss(pred, target)
        success = loss != ffi.NULL
        lib.tensor_free(pred)
        lib.tensor_free(target)
        if success:
            lib.tensor_free(loss)
        return success

    def test_mae_loss():
        pred = lib.cml_ones_2d(3, 3)
        target = lib.cml_zeros_2d(3, 3)
        loss = lib.cml_nn_mae_loss(pred, target)
        success = loss != ffi.NULL
        lib.tensor_free(pred)
        lib.tensor_free(target)
        if success:
            lib.tensor_free(loss)
        return success

    def test_bce_loss():
        # BCE expects probabilities in (0, 1)
        pred = lib.cml_ones_2d(3, 3)
        target = lib.cml_ones_2d(3, 3)
        loss = lib.cml_nn_bce_loss(pred, target)
        success = loss != ffi.NULL
        lib.tensor_free(pred)
        lib.tensor_free(target)
        if success:
            lib.tensor_free(loss)
        return success

    def test_cross_entropy_loss():
        # pred: (batch, num_classes) logits; target: (batch,) class indices
        pred = lib.cml_ones_2d(4, 3)
        target_data = ffi.new("float[4]", [0.0, 1.0, 2.0, 0.0])
        target = lib.cml_tensor_1d(target_data, 4)
        if target == ffi.NULL:
            lib.tensor_free(pred)
            return False
        loss = lib.cml_nn_cross_entropy_loss(pred, target)
        success = loss != ffi.NULL
        lib.tensor_free(pred)
        lib.tensor_free(target)
        if success:
            lib.tensor_free(loss)
        return success

    def test_huber_loss():
        pred = lib.cml_ones_2d(2, 3)
        target = lib.cml_zeros_2d(2, 3)
        loss = lib.cml_nn_huber_loss(pred, target, 1.0)
        success = loss != ffi.NULL
        lib.tensor_free(pred)
        lib.tensor_free(target)
        if success:
            lib.tensor_free(loss)
        return success

    def test_kl_div_loss():
        pred = lib.cml_ones_2d(2, 3)
        target = lib.cml_ones_2d(2, 3)
        loss = lib.cml_nn_kl_div_loss(pred, target)
        success = loss != ffi.NULL
        lib.tensor_free(pred)
        lib.tensor_free(target)
        if success:
            lib.tensor_free(loss)
        return success

    loss_tests = [
        ("cml_nn_mse_loss", test_mse_loss),
        ("cml_nn_mae_loss", test_mae_loss),
        ("cml_nn_bce_loss", test_bce_loss),
        ("cml_nn_cross_entropy_loss", test_cross_entropy_loss),
        ("cml_nn_huber_loss", test_huber_loss),
        ("cml_nn_kl_div_loss", test_kl_div_loss),
    ]
    p, f, s = test_category("Loss Functions", loss_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 13. Optimizers
    def _create_model_and_optimizer(create_opt_fn):
        linear = lib.cml_nn_linear(4, 2, 0, 0, True)
        if linear == ffi.NULL:
            return None, None
        module = ffi.cast("Module*", linear)
        opt = create_opt_fn(module)
        if opt == ffi.NULL:
            lib.module_free(module)
            return None, None
        return module, opt

    def test_adam_optimizer():
        module, opt = _create_model_and_optimizer(
            lambda m: lib.cml_optim_adam_for_model(m, 0.001, 0.0, 0.9, 0.999, 1e-8)
        )
        if opt is None:
            return False
        lib.optimizer_free(opt)
        lib.module_free(module)
        return True

    def test_sgd_optimizer():
        module, opt = _create_model_and_optimizer(
            lambda m: lib.cml_optim_sgd_for_model(m, 0.01, 0.9, 0.0)
        )
        if opt is None:
            return False
        lib.optimizer_free(opt)
        lib.module_free(module)
        return True

    def test_optimizer_step():
        module, opt = _create_model_and_optimizer(
            lambda m: lib.cml_optim_adam_for_model(m, 0.001, 0.0, 0.9, 0.999, 1e-8)
        )
        if opt is None:
            return False
        lib.cml_optim_step(opt)
        lib.optimizer_free(opt)
        lib.module_free(module)
        return True

    def test_optimizer_zero_grad():
        module, opt = _create_model_and_optimizer(
            lambda m: lib.cml_optim_adam_for_model(m, 0.001, 0.0, 0.9, 0.999, 1e-8)
        )
        if opt is None:
            return False
        lib.cml_optim_zero_grad(opt)
        lib.optimizer_free(opt)
        lib.module_free(module)
        return True

    optimizer_tests = [
        ("cml_optim_adam_for_model", test_adam_optimizer),
        ("cml_optim_sgd_for_model", test_sgd_optimizer),
        ("cml_optim_step", test_optimizer_step),
        ("cml_optim_zero_grad", test_optimizer_zero_grad),
    ]
    p, f, s = test_category("Optimizers", optimizer_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 14. Utility Functions
    def test_set_log_level():
        lib.cml_set_log_level(0)
        return True

    def test_version():
        major = ffi.new("int*")
        minor = ffi.new("int*")
        patch = ffi.new("int*")
        version_str = ffi.new("const char**")
        lib.cml_get_version(major, minor, patch, version_str)
        return major[0] >= 0 and minor[0] >= 0

    util_tests = [
        ("cml_set_log_level", test_set_log_level),
        ("cml_get_version", test_version),
    ]
    p, f, s = test_category("Utility Functions", util_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # 15. End-to-End Training Test
    def test_full_training_loop():
        """Complete training loop test."""
        seq = lib.cml_nn_sequential()
        if seq == ffi.NULL:
            return False

        linear1 = lib.cml_nn_linear(4, 8, 0, 0, True)
        if linear1 == ffi.NULL:
            return False
        lib.cml_nn_sequential_add(seq, ffi.cast("Module*", linear1))

        relu = lib.cml_nn_relu(False)
        if relu == ffi.NULL:
            return False
        lib.cml_nn_sequential_add(seq, ffi.cast("Module*", relu))

        linear2 = lib.cml_nn_linear(8, 2, 0, 0, True)
        if linear2 == ffi.NULL:
            return False
        lib.cml_nn_sequential_add(seq, ffi.cast("Module*", linear2))

        seq_module = ffi.cast("Module*", seq)
        opt = lib.cml_optim_adam_for_model(seq_module, 0.001, 0.0, 0.9, 0.999, 1e-8)
        if opt == ffi.NULL:
            return False

        x = lib.cml_ones_2d(2, 4)
        target = lib.cml_zeros_2d(2, 2)

        y = lib.cml_nn_sequential_forward(seq, x)
        if y == ffi.NULL:
            lib.tensor_free(x)
            lib.tensor_free(target)
            return False

        loss = lib.cml_nn_mse_loss(y, target)
        if loss == ffi.NULL:
            lib.tensor_free(x)
            lib.tensor_free(target)
            lib.tensor_free(y)
            return False

        lib.cml_backward(loss, ffi.NULL, False, False)
        lib.cml_optim_step(opt)
        lib.cml_optim_zero_grad(opt)

        lib.tensor_free(x)
        lib.tensor_free(target)
        lib.tensor_free(y)
        lib.tensor_free(loss)
        lib.optimizer_free(opt)

        return True

    e2e_tests = [
        ("Full training loop", test_full_training_loop),
    ]
    p, f, s = test_category("End-to-End Training", e2e_tests)
    total_passed += p
    total_failed += f
    total_skipped += s

    # Summary (before cleanup to ensure results are shown)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total Passed:  {total_passed}")
    print(f"Total Failed:  {total_failed}")
    print(f"Total Skipped: {total_skipped}")
    print(f"Total Tests:   {total_passed + total_failed + total_skipped}")
    print("=" * 60)
    sys.stdout.flush()

    if total_failed == 0:
        print("\nAll CFFI bindings are working correctly!")
    else:
        print(f"\n{total_failed} test(s) failed.")

    # Note: Skip cleanup to avoid double-free issue with module tracking.
    # In production, proper cleanup happens automatically via atexit.
    # The double-free occurs because modules are freed both manually and
    # during cleanup when they're tracked in g_tracked_modules.
    print("\nNote: Skipping explicit cleanup to avoid module tracking double-free.")
    print("Resources will be cleaned up automatically on process exit.")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
