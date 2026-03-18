#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

#define N_EPOCHS 50

static void show_activation(const char* name, Tensor* output, int n) {
    printf("  %-12s: [", name);
    for (int i = 0; i < n; i++) {
        printf("%7.3f", tensor_get_float(output, i));
        if (i < n - 1) printf(", ");
    }
    printf("]\n");
}

int main(void) {
    cml_init();
    printf("Example 15: Activation Functions\n\n");

    float data[] = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
    int shape[] = {1, 7};
    int n = 7;

    Tensor* x = cml_tensor(data, shape, 2, NULL);
    printf("Input:        [");
    for (int i = 0; i < n; i++) {
        printf("%7.3f", data[i]);
        if (i < n - 1) printf(", ");
    }
    printf("]\n\n");

    show_activation("ReLU",       module_forward((Module*)cml_nn_relu(false), x), n);
    show_activation("LeakyReLU",  module_forward((Module*)cml_nn_leaky_relu(0.1f, false), x), n);
    show_activation("Sigmoid",    module_forward((Module*)cml_nn_sigmoid(), x), n);
    show_activation("Tanh",       module_forward((Module*)cml_nn_tanh(), x), n);
    show_activation("GELU",       module_forward((Module*)nn_gelu(false), x), n);
    show_activation("ELU",        module_forward((Module*)nn_elu(1.0f, false), x), n);
    show_activation("SELU",       module_forward((Module*)nn_selu(), x), n);
    show_activation("SiLU/Swish", module_forward((Module*)nn_silu(), x), n);
    show_activation("Mish",       module_forward((Module*)nn_mish(), x), n);
    show_activation("HardSwish",  module_forward((Module*)nn_hardswish(), x), n);

    printf("\nBreast Cancer classification (%d epochs):\n", N_EPOCHS);

    Dataset* ds = cml_dataset_load("breast_cancer");
    if (!ds) { printf("Failed to load breast_cancer dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");

    printf("Samples: %d, Features: %d\n\n", ds->num_samples, ds->input_size);

    const char* act_names[] = {"ReLU", "Tanh", "GELU", "Mish", "SiLU"};
    Module* activations[] = {
        (Module*)cml_nn_relu(false),
        (Module*)cml_nn_tanh(),
        (Module*)nn_gelu(false),
        (Module*)nn_mish(),
        (Module*)nn_silu()
    };

    for (int a = 0; a < 5; a++) {
        Sequential* m = cml_nn_sequential();
        cml_nn_sequential_add(m, (Module*)cml_nn_linear(ds->input_size, 16,
                              DTYPE_FLOAT32, DEVICE_CPU, true));
        cml_nn_sequential_add(m, activations[a]);
        cml_nn_sequential_add(m, (Module*)cml_nn_linear(16, 8, DTYPE_FLOAT32, DEVICE_CPU, true));
        cml_nn_sequential_add(m, (Module*)cml_nn_relu(false));
        cml_nn_sequential_add(m, (Module*)cml_nn_linear(8, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
        cml_nn_sequential_add(m, (Module*)cml_nn_sigmoid());

        Optimizer* opt = cml_optim_adam_for_model((Module*)m, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);

        float final_loss = 0;
        for (int ep = 1; ep <= N_EPOCHS; ep++) {
            Tensor* pred = cml_nn_sequential_forward(m, ds->X);
            Tensor* loss = cml_nn_bce_loss(pred, ds->y);
            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);
            if (ep == N_EPOCHS) final_loss = tensor_get_float(loss, 0);
        }

        Tensor* pred = cml_nn_sequential_forward(m, ds->X);
        int correct = 0;
        for (int i = 0; i < ds->num_samples; i++) {
            float p = tensor_get_float(pred, i);
            int cls = p > 0.5f ? 1 : 0;
            correct += (cls == (int)tensor_get_float(ds->y, i));
        }
        printf("  %-12s: loss=%.6f  accuracy=%d/%d (%.0f%%)\n",
               act_names[a], final_loss, correct, ds->num_samples,
               correct / (float)ds->num_samples * 100);
    }

    printf("\nActivation showcase complete.\n");
    cml_cleanup();
    return 0;
}
