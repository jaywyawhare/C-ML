/**
 * Example 11: GAN (Generative Adversarial Network)
 *
 * Generator:     noise(64) -> 128 -> 64 (fake digit vectors)
 * Discriminator: input(64) -> 128 -> 1 (real/fake probability)
 *
 * Trains G to produce digit-like 64-dim vectors from the Digits 8x8 dataset.
 */
#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

#define NOISE_DIM  64
#define DATA_DIM   64
#define HIDDEN_DIM 128
#define BATCH_SIZE 32
#define N_EPOCHS   100

static void fill_noise(float* buf, int n) {
    for (int i = 0; i < n; i++)
        buf[i] = (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;
}

int main(void) {
    cml_init();
    srand(42);
    printf("Example 11: GAN (Digits 8x8)\n\n");

    Dataset* ds = cml_dataset_load("digits");
    if (!ds) { printf("Failed to load digits dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");
    float* real_pool = (float*)tensor_data_ptr(ds->X);
    int pool_size = ds->num_samples;

    printf("Real data: %d samples, %d-dim\n\n", pool_size, DATA_DIM);

    /* Generator: noise -> hidden -> data */
    Sequential* G = cml_nn_sequential();
    cml_nn_sequential_add(G, (Module*)cml_nn_linear(NOISE_DIM, HIDDEN_DIM, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(G, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(G, (Module*)cml_nn_linear(HIDDEN_DIM, DATA_DIM, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(G, (Module*)cml_nn_sigmoid());

    /* Discriminator: data -> hidden -> probability */
    Sequential* D = cml_nn_sequential();
    cml_nn_sequential_add(D, (Module*)cml_nn_linear(DATA_DIM, HIDDEN_DIM, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(D, (Module*)cml_nn_relu(false));
    cml_nn_sequential_add(D, (Module*)cml_nn_linear(HIDDEN_DIM, 1, DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(D, (Module*)cml_nn_sigmoid());

    printf("Generator:\n");
    cml_summary((Module*)G);
    printf("Discriminator:\n");
    cml_summary((Module*)D);

    Optimizer* opt_G = cml_optim_adam_for_model((Module*)G, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    Optimizer* opt_D = cml_optim_adam_for_model((Module*)D, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);

    int data_shape[]  = {BATCH_SIZE, DATA_DIM};
    int label_shape[] = {BATCH_SIZE, 1};
    float ones[BATCH_SIZE], zeros[BATCH_SIZE];
    for (int i = 0; i < BATCH_SIZE; i++) { ones[i] = 1.0f; zeros[i] = 0.0f; }

    float real_buf[BATCH_SIZE * DATA_DIM];
    float noise_buf[BATCH_SIZE * NOISE_DIM];

    for (int epoch = 1; epoch <= N_EPOCHS; epoch++) {
        /* Sample random real batch from digits dataset */
        for (int i = 0; i < BATCH_SIZE; i++) {
            int idx = rand() % pool_size;
            for (int j = 0; j < DATA_DIM; j++)
                real_buf[i * DATA_DIM + j] = real_pool[idx * DATA_DIM + j];
        }

        /* Train Discriminator */
        Tensor* real = cml_tensor(real_buf, data_shape, 2, NULL);
        Tensor* d_real = cml_nn_sequential_forward(D, real);
        Tensor* label_real = cml_tensor(ones, label_shape, 2, NULL);
        Tensor* loss_d_real = cml_nn_bce_loss(d_real, label_real);

        fill_noise(noise_buf, BATCH_SIZE * NOISE_DIM);
        Tensor* noise = cml_tensor(noise_buf, data_shape, 2, NULL);
        Tensor* fake = cml_nn_sequential_forward(G, noise);
        Tensor* d_fake = cml_nn_sequential_forward(D, fake);
        Tensor* label_fake = cml_tensor(zeros, label_shape, 2, NULL);
        Tensor* loss_d_fake = cml_nn_bce_loss(d_fake, label_fake);

        Tensor* loss_d = cml_add(loss_d_real, loss_d_fake);

        cml_optim_zero_grad(opt_D);
        cml_backward(loss_d, NULL, false, false);
        cml_optim_step(opt_D);

        /* Train Generator */
        fill_noise(noise_buf, BATCH_SIZE * NOISE_DIM);
        noise = cml_tensor(noise_buf, data_shape, 2, NULL);
        fake = cml_nn_sequential_forward(G, noise);
        Tensor* d_out = cml_nn_sequential_forward(D, fake);
        Tensor* loss_g = cml_nn_bce_loss(d_out, label_real);

        cml_optim_zero_grad(opt_G);
        cml_backward(loss_g, NULL, false, false);
        cml_optim_step(opt_G);

        if (epoch % 20 == 0) {
            tensor_ensure_executed(loss_d);
            tensor_ensure_executed(loss_g);
            printf("Epoch %3d  D_loss: %.4f  G_loss: %.4f\n",
                   epoch, tensor_get_float(loss_d, 0), tensor_get_float(loss_g, 0));
        }
    }

    /* Generate final samples */
    printf("\nGenerated samples (first 8 pixels of 4 samples):\n");
    fill_noise(noise_buf, BATCH_SIZE * NOISE_DIM);
    Tensor* noise_final = cml_tensor(noise_buf, data_shape, 2, NULL);
    Tensor* generated = cml_nn_sequential_forward(G, noise_final);
    for (int i = 0; i < 4; i++) {
        printf("  [");
        for (int j = 0; j < 8; j++) {
            printf("%.3f", tensor_get_float(generated, i * DATA_DIM + j));
            if (j < 7) printf(", ");
        }
        printf(", ...]\n");
    }

    cml_cleanup();
    return 0;
}
