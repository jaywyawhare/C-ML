#include "cml.h"
#include <stdio.h>

#define HIDDEN_SIZE 16
#define TRAIN_RATIO 0.75f

int main(void) {
    cml_init();
    printf("Example 08: LSTM Time Series (Airline Passengers)\n\n");

    Dataset* ds = cml_dataset_load("airline");
    if (!ds) { printf("Failed to load airline dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");

    int total = ds->num_samples;
    int train_steps = (int)(total * TRAIN_RATIO);
    if (train_steps > 30) train_steps = 30; /* Cap for speed */
    int test_steps = total - train_steps - 1;
    if (test_steps > 20) test_steps = 20;
    float* series = (float*)tensor_data_ptr(ds->X);

    printf("Total: %d steps, Train: %d, Test: %d\n\n", total, train_steps, test_steps);

    int x_shape[] = {1, 1};
    int h_shape[] = {1, HIDDEN_SIZE};
    int y_shape[] = {1, 1};

    LSTMCell* lstm = cml_nn_lstm_cell(1, HIDDEN_SIZE, true, DTYPE_FLOAT32, DEVICE_CPU);
    Linear* fc = cml_nn_linear(HIDDEN_SIZE, 1, DTYPE_FLOAT32, DEVICE_CPU, true);

    printf("LSTM: input=1, hidden=%d\n\n", HIDDEN_SIZE);

    Parameter* params[6];
    int np = 0;
    params[np++] = lstm->weight_ih;
    params[np++] = lstm->weight_hh;
    params[np++] = lstm->bias_ih;
    params[np++] = lstm->bias_hh;
    params[np++] = linear_get_weight(fc);
    params[np++] = linear_get_bias(fc);

    Optimizer* opt = cml_optim_adam(params, np, 0.001f, 0.0f, 0.9f, 0.999f, 1e-8f);
    optimizer_set_grad_clip_norm(opt, 1.0f);

    for (int epoch = 1; epoch <= 50; epoch++) {
        float total_loss = 0;
        Tensor* h = cml_zeros(h_shape, 2, NULL);
        Tensor* c = cml_zeros(h_shape, 2, NULL);

        for (int t = 0; t < train_steps - 1; t++) {
            Tensor* x_t = cml_tensor(&series[t], x_shape, 2, NULL);
            float y_val = series[t + 1];
            Tensor* y_t = cml_tensor(&y_val, y_shape, 2, NULL);

            Tensor* h_out = NULL;
            Tensor* c_out = NULL;
            lstm_cell_forward(lstm, x_t, h, c, &h_out, &c_out);
            h = cml_detach(h_out);
            c = cml_detach(c_out);

            Tensor* pred = linear_forward((Module*)fc, h);
            Tensor* loss = cml_nn_mse_loss(pred, y_t);

            cml_optim_zero_grad(opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(opt);

            total_loss += tensor_get_float(loss, 0);
        }

        if (epoch % 10 == 0)
            printf("Epoch %3d  Avg Loss: %.6f\n", epoch, total_loss / (train_steps - 1));
    }

    printf("\nTest predictions:\n");
    Tensor* h = cml_zeros(h_shape, 2, NULL);
    Tensor* c_state = cml_zeros(h_shape, 2, NULL);

    for (int t = 0; t < train_steps; t++) {
        Tensor* x_t = cml_tensor(&series[t], x_shape, 2, NULL);
        Tensor* h_out = NULL;
        Tensor* c_out = NULL;
        lstm_cell_forward(lstm, x_t, h, c_state, &h_out, &c_out);
        h = h_out;
        c_state = c_out;
    }

    float test_mse = 0;
    for (int t = train_steps; t < total - 1; t++) {
        Tensor* x_t = cml_tensor(&series[t], x_shape, 2, NULL);
        Tensor* h_out = NULL;
        Tensor* c_out = NULL;
        lstm_cell_forward(lstm, x_t, h, c_state, &h_out, &c_out);
        h = h_out;
        c_state = c_out;

        Tensor* pred = linear_forward((Module*)fc, h);
        float p = tensor_get_float(pred, 0);
        float y_val = series[t + 1];
        float err = p - y_val;
        test_mse += err * err;
        if (t < train_steps + 10)
            printf("  t=%3d: pred=%.4f target=%.4f\n", t + 1, p, y_val);
    }
    printf("Test MSE: %.6f (on %d steps)\n", test_mse / test_steps, test_steps);

    cml_cleanup();
    return 0;
}
