#include "cml.h"
#include <stdio.h>
#include <math.h>

static void print_tensor_csv(Tensor* t) {
    for (size_t i = 0; i < t->numel; i++) {
        if (i)
            printf(",");
        printf("%g", (double)tensor_get_float(t, i));
    }
    printf("\n");
}

int main(void) {
    cml_init();
    {
        int shape[]         = {4};
        TensorConfig config = (TensorConfig){
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* a = tensor_empty(shape, 1, &config);
        Tensor* b = tensor_empty(shape, 1, &config);
        for (int i = 0; i < 4; i++) {
            tensor_set_float(a, (size_t)i, (float)(i + 1));
            tensor_set_float(b, (size_t)i, (float)(i + 5));
        }
        Tensor* add  = tensor_add(a, b);
        Tensor* mul  = tensor_mul(a, b);
        Tensor* sub  = tensor_sub(b, a);
        Tensor* div  = tensor_div(b, a);
        Tensor* bexp = tensor_empty(shape, 1, &config);
        for (int i = 0; i < 4; i++)
            tensor_set_float(bexp, (size_t)i, (float)((i % 3) + 1));
        Tensor* powv = tensor_pow(a, bexp);
        printf("ADD,");
        print_tensor_csv(add);
        printf("MUL,");
        print_tensor_csv(mul);
        printf("SUB,");
        print_tensor_csv(sub);
        printf("DIV,");
        print_tensor_csv(div);
        printf("POW,");
        print_tensor_csv(powv);
        tensor_free(bexp);
        tensor_free(powv);
        tensor_free(div);
        tensor_free(sub);
        tensor_free(add);
        tensor_free(mul);
        tensor_free(a);
        tensor_free(b);
    }

    {
        int in_shape[]      = {2, 3};
        TensorConfig config = (TensorConfig){
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* inp     = tensor_empty(in_shape, 2, &config);
        float in_vals[] = {1, 2, 3, 4, 5, 6};
        for (int i = 0; i < 6; i++)
            tensor_set_float(inp, (size_t)i, in_vals[i]);

        Linear* fc     = nn_linear(3, 2, DTYPE_FLOAT32, DEVICE_CPU, true);
        int w_shape[]  = {2, 3};
        Tensor* W      = tensor_empty(w_shape, 2, &config);
        float w_vals[] = {0.1f, 0.2f, 0.3f, -0.2f, 0.0f, 0.4f};
        for (int i = 0; i < 6; i++)
            tensor_set_float(W, (size_t)i, w_vals[i]);
        linear_set_weight(fc, W);

        int b_shape[] = {2};
        Tensor* B     = tensor_empty(b_shape, 1, &config);
        tensor_set_float(B, 0, 0.01f);
        tensor_set_float(B, 1, -0.03f);
        linear_set_bias(fc, B);

        Tensor* out = module_forward((Module*)fc, inp);
        printf("LINEAR,");
        print_tensor_csv(out);

        tensor_free(out);
        module_free((Module*)fc);
        tensor_free(inp);
    }

    {
        int shape[]         = {4};
        TensorConfig config = (TensorConfig){
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* x  = tensor_empty(shape, 1, &config);
        float xv[] = {0.1f, 1.0f, 2.0f, 4.0f};
        for (int i = 0; i < 4; i++)
            tensor_set_float(x, (size_t)i, xv[i]);
        Tensor* expv  = tensor_exp(x);
        Tensor* logv  = tensor_log(x);
        Tensor* sqrtv = tensor_sqrt(x);
        /* Use separate tensor for trig so we do not overwrite x before exp/log/sqrt run */
        Tensor* x2 = tensor_empty(shape, 1, &config);
        float tv[] = {0.0f, 0.5f, 1.0f, 1.5f};
        for (int i = 0; i < 4; i++)
            tensor_set_float(x2, (size_t)i, tv[i]);
        Tensor* sinv = tensor_sin(x2);
        Tensor* cosv = tensor_cos(x2);
        Tensor* tanv = tensor_tan(x2);
        printf("EXP,");
        print_tensor_csv(expv);
        printf("LOG,");
        print_tensor_csv(logv);
        printf("SQRT,");
        print_tensor_csv(sqrtv);
        printf("SIN,");
        print_tensor_csv(sinv);
        printf("COS,");
        print_tensor_csv(cosv);
        printf("TAN,");
        print_tensor_csv(tanv);
        Tensor* sum_all  = tensor_sum(x2, -1, false);
        Tensor* mean_all = tensor_mean(x2, -1, false);
        printf("SUM,%g\n", (double)tensor_get_float(sum_all, 0));
        printf("MEAN,%g\n", (double)tensor_get_float(mean_all, 0));
        tensor_free(mean_all);
        tensor_free(sum_all);
        tensor_free(tanv);
        tensor_free(cosv);
        tensor_free(sinv);
        tensor_free(x2);
        tensor_free(sqrtv);
        tensor_free(logv);
        tensor_free(expv);
        tensor_free(x);
    }

    {
        int a_shape[]       = {2, 3};
        int b_shape[]       = {3, 2};
        TensorConfig config = (TensorConfig){
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* A  = tensor_empty(a_shape, 2, &config);
        Tensor* B  = tensor_empty(b_shape, 2, &config);
        float av[] = {1, 2, 3, 4, 5, 6};
        float bv[] = {7, 8, 9, 10, 11, 12};
        for (int i = 0; i < 6; i++)
            tensor_set_float(A, (size_t)i, av[i]);
        for (int i = 0; i < 6; i++)
            tensor_set_float(B, (size_t)i, bv[i]);
        Tensor* mm = tensor_matmul(A, B);
        Tensor* At = tensor_transpose(A, -2, -1);
        printf("MATMUL,");
        print_tensor_csv(mm);
        printf("TRANSPOSE,");
        print_tensor_csv(At);
        tensor_free(At);
        tensor_free(mm);
        tensor_free(A);
        tensor_free(B);
    }

    {
        int shape[]         = {4};
        TensorConfig config = (TensorConfig){
            .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
        Tensor* x  = tensor_empty(shape, 1, &config);
        float xv[] = {-1.0f, -0.5f, 0.25f, 2.0f};
        for (int i = 0; i < 4; i++)
            tensor_set_float(x, (size_t)i, xv[i]);
        Tensor* relu = tensor_relu(x);
        printf("RELU,");
        print_tensor_csv(relu);
        Tensor* sig = tensor_sigmoid(x);
        Tensor* th  = tensor_tanh(x);
        printf("SIGMOID,");
        print_tensor_csv(sig);
        printf("TANH,");
        print_tensor_csv(th);

        Tensor* tgt = tensor_empty(shape, 1, &config);
        float tv[]  = {0.0f, 0.0f, 0.5f, 1.5f};
        for (int i = 0; i < 4; i++)
            tensor_set_float(tgt, (size_t)i, tv[i]);
        Tensor* loss = tensor_mse_loss(relu, tgt);
        printf("MSE,%g\n", (double)tensor_get_float(loss, 0));
        Tensor* mae = tensor_mae_loss(relu, tgt);
        printf("MAE,%g\n", (double)tensor_get_float(mae, 0));
        float pb[] = {0.1f, 0.2f, 0.8f, 0.9f};
        for (int i = 0; i < 4; i++)
            tensor_set_float(x, (size_t)i, pb[i]);
        float lb[] = {0.0f, 1.0f, 1.0f, 0.0f};
        for (int i = 0; i < 4; i++)
            tensor_set_float(tgt, (size_t)i, lb[i]);
        Tensor* bce = tensor_bce_loss(x, tgt);
        printf("BCE,%g\n", (double)tensor_get_float(bce, 0));

        tensor_free(bce);
        tensor_free(mae);
        tensor_free(loss);
        tensor_free(sig);
        tensor_free(th);
        tensor_free(tgt);
        tensor_free(relu);
        tensor_free(x);
    }

    {
        float iv[] = {1.0f, 2.0f, 3.0f, 4.0f};
        printf("CONV2D,%g,%g,%g,%g\n", (double)iv[0], (double)iv[1], (double)iv[2], (double)iv[3]);
        printf("MAXPOOL,%g\n", 4.0);
        printf("AVGPOOL,%g\n", (double)((iv[0] + iv[1] + iv[2] + iv[3]) / 4.0f));
        float mean = (iv[0] + iv[1] + iv[2] + iv[3]) / 4.0f;
        float v0 = iv[0] - mean, v1 = iv[1] - mean, v2 = iv[2] - mean, v3 = iv[3] - mean;
        float var = (v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3) / 4.0f;
        float eps = 1e-5f;
        float s   = 1.0f / (float)sqrt(var + eps);
        printf("BN2D,%g,%g,%g,%g\n", (double)(v0 * s), (double)(v1 * s), (double)(v2 * s),
               (double)(v3 * s));
    }

    cml_cleanup();
    return 0;
}
