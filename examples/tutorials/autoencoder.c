#include "cml.h"
#include <stdio.h>

#define LATENT_DIM 8

int main(void) {
    cml_init();
    printf("Example 05: Autoencoder (Digits 8x8)\n\n");

    Dataset* ds = cml_dataset_load("digits");
    if (!ds) { printf("Failed to load digits dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");

    int n = 200;
    if (n > ds->num_samples) n = ds->num_samples;
    int feat = ds->input_size;

    printf("Using %d/%d samples, Features: %d, Latent: %d\n\n",
           n, ds->num_samples, feat, LATENT_DIM);

    int sub_shape[] = {n, feat};
    Tensor* X = cml_tensor(tensor_data_ptr(ds->X), sub_shape, 2, NULL);

    Sequential* autoenc = cml_nn_sequential();
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_linear(64, 32, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_linear(32, LATENT_DIM, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_linear(LATENT_DIM, 32, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_linear(32, 64, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(autoenc, (Module*)cml_nn_sigmoid());

    printf("Autoencoder:\n");
    cml_summary((Module*)autoenc);

    Optimizer* opt = cml_optim_adam_for_model((Module*)autoenc, 0.005f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 100; epoch++) {
        Tensor* recon = cml_nn_sequential_forward(autoenc, X);
        Tensor* loss = cml_nn_mse_loss(recon, X);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);
        cml_reset_ir_context();

        if (epoch % 20 == 0)
            printf("Epoch %4d  Reconstruction Loss: %.6f\n", epoch, tensor_get_float(loss, 0));
    }

    printf("\nReconstructed vs original (first 5, showing 8 pixels):\n");
    Tensor* recon = cml_nn_sequential_forward(autoenc, X);
    for (int i = 0; i < 5; i++) {
        float label = tensor_get_float(ds->y, i);
        printf("  digit=%d orig: ", (int)label);
        for (int j = 0; j < 8; j++)
            printf("%.2f ", tensor_get_float(X, i * 64 + j));
        printf("\n           reco: ");
        for (int j = 0; j < 8; j++)
            printf("%.2f ", tensor_get_float(recon, i * 64 + j));
        printf("\n");
    }

    cml_cleanup();
    return 0;
}
