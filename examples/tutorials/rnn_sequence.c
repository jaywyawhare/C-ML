#include "cml.h"
#include <stdio.h>
#include <string.h>

#define HIDDEN_SIZE 16
#define SEQ_LEN     5

int main(void) {
    cml_init();
    printf("Example 07: RNN Sequence Processing (Airline)\n\n");

    Dataset* ds = cml_dataset_load("airline");
    if (!ds) { printf("Failed to load airline dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");

    int total = ds->num_samples;
    int n_seqs = total - SEQ_LEN;
    float* seqs = (float*)tensor_data_ptr(ds->X);

    printf("Total timesteps: %d, Sequences: %d (window=%d)\n\n", total, n_seqs, SEQ_LEN);

    int step_shape[] = {1, 1};
    int h_shape[]    = {1, HIDDEN_SIZE};
    int y_shape[]    = {1, 1};

    RNNCell* rnn = cml_nn_rnn_cell(1, HIDDEN_SIZE, true, DTYPE_FLOAT32, DEVICE_CPU);
    Linear* fc = cml_nn_linear(HIDDEN_SIZE, 1, DTYPE_FLOAT32, DEVICE_CPU, true);

    printf("RNN: input=1, hidden=%d\n", HIDDEN_SIZE);
    printf("Output: %d -> 1\n\n", HIDDEN_SIZE);

    Parameter* params[6];
    int np = 0;
    params[np++] = rnn->weight_ih;
    params[np++] = rnn->weight_hh;
    params[np++] = rnn->bias_ih;
    params[np++] = rnn->bias_hh;
    params[np++] = linear_get_weight(fc);
    params[np++] = linear_get_bias(fc);

    Optimizer* opt = cml_optim_adam(params, np, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    optimizer_set_grad_clip_norm(opt, 1.0f);

    int train_seqs = n_seqs * 3 / 4;
    if (train_seqs > 20) train_seqs = 20; /* Keep fast for demo */

    for (int epoch = 1; epoch <= 50; epoch++) {
        float total_loss = 0;
        for (int s = 0; s < train_seqs; s++) {
            Tensor* h = cml_zeros(h_shape, 2, NULL);
            for (int t = 0; t < SEQ_LEN; t++) {
                Tensor* x_t = cml_tensor(&seqs[s + t], step_shape, 2, NULL);
                Tensor* h_new = rnn_cell_forward(rnn, x_t, h);
                h = cml_detach(h_new);
            }
            Tensor* out = linear_forward((Module*)fc, h);
            float target = seqs[s + SEQ_LEN];
            Tensor* y_t = cml_tensor(&target, y_shape, 2, NULL);
            Tensor* loss = cml_nn_mse_loss(out, y_t);

            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);
            total_loss += tensor_get_float(loss, 0);
            cml_reset_ir_context();
        }

        if (epoch % 10 == 0)
            printf("Epoch %3d  Avg Loss: %.6f\n", epoch, total_loss / train_seqs);
    }

    printf("\nPredictions (last 10 test sequences):\n");
    float test_mse = 0;
    int test_count = 0;
    for (int s = train_seqs; s < n_seqs && test_count < 10; s++, test_count++) {
        Tensor* h = cml_zeros(h_shape, 2, NULL);
        for (int t = 0; t < SEQ_LEN; t++) {
            Tensor* x_t = cml_tensor(&seqs[s + t], step_shape, 2, NULL);
            Tensor* h_new = rnn_cell_forward(rnn, x_t, h);
            h = cml_detach(h_new);
        }
        Tensor* out = linear_forward((Module*)fc, h);
        float p = tensor_get_float(out, 0);
        float target = seqs[s + SEQ_LEN];
        float err = p - target;
        test_mse += err * err;
        printf("  seq %d: pred=%.3f target=%.3f\n", s, p, target);
    }
    if (test_count > 0)
        printf("Test MSE: %.6f\n", test_mse / test_count);

    cml_cleanup();
    return 0;
}
