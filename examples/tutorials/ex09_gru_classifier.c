/**
 * Example 09: GRU Sequence Classifier
 *
 * Classifies Iris samples by feeding 4 features as a 4-step sequence into a GRU.
 * Dataset: Iris (150 samples, 4 features as 4 timesteps, 3 classes).
 */
#include "cml.h"
#include <stdio.h>

#define INPUT_SIZE  1
#define HIDDEN_SIZE 16
#define SEQ_LEN     4   /* 4 features = 4 timesteps */

int main(void) {
    cml_init();
    printf("Example 09: GRU Sequence Classifier (Iris)\n\n");

    Dataset* ds = cml_dataset_load("iris");
    if (!ds) { printf("Failed to load iris dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");
    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    printf("Train: %d, Test: %d (4 features as 4-step sequence)\n\n",
           train->num_samples, test->num_samples);

    int x_shape[] = {1, INPUT_SIZE};
    int h_shape[] = {1, HIDDEN_SIZE};
    int y_shape[] = {1, 1};

    GRUCell* gru = cml_nn_gru_cell(INPUT_SIZE, HIDDEN_SIZE, true, DTYPE_FLOAT32, DEVICE_CPU);
    Sequential* head = cml_nn_sequential();
    cml_nn_sequential_add(head, (Module*)cml_nn_linear(HIDDEN_SIZE, 1,
                          DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(head, (Module*)cml_nn_sigmoid());

    printf("GRU: input=%d, hidden=%d\n\n", INPUT_SIZE, HIDDEN_SIZE);

    Parameter* params[4];
    int np = 0;
    params[np++] = gru->weight_ih;
    params[np++] = gru->weight_hh;
    params[np++] = gru->bias_ih;
    params[np++] = gru->bias_hh;

    Optimizer* gru_opt = cml_optim_adam(params, np, 0.005f, 0.0f, 0.9f, 0.999f, 1e-8f);
    Optimizer* head_opt = cml_optim_adam_for_model((Module*)head, 0.005f, 0.0f, 0.9f, 0.999f, 1e-8f);

    /* Simplify to binary: class 0 vs classes 1+2 */
    float* train_X = (float*)tensor_data_ptr(train->X);
    float* train_y = (float*)tensor_data_ptr(train->y);
    float* test_X = (float*)tensor_data_ptr(test->X);
    float* test_y = (float*)tensor_data_ptr(test->y);

    for (int epoch = 1; epoch <= 50; epoch++) {
        float total_loss = 0;
        for (int s = 0; s < train->num_samples; s++) {
            Tensor* h = cml_zeros(h_shape, 2, NULL);
            for (int t = 0; t < SEQ_LEN; t++) {
                float val = train_X[s * 4 + t];
                Tensor* x_t = cml_tensor(&val, x_shape, 2, NULL);
                h = gru_cell_forward(gru, x_t, h);
            }
            Tensor* pred = cml_nn_sequential_forward(head, h);
            float label = (train_y[s] > 0.5f) ? 1.0f : 0.0f;
            Tensor* y = cml_tensor(&label, y_shape, 2, NULL);
            Tensor* loss = cml_nn_bce_loss(pred, y);

            cml_optim_zero_grad(gru_opt);
            cml_optim_zero_grad(head_opt);
            cml_backward(loss, NULL, false, false);
            cml_optim_step(gru_opt);
            cml_optim_step(head_opt);

            total_loss += tensor_get_float(loss, 0);
        }
        if (epoch % 10 == 0)
            printf("Epoch %3d  Avg Loss: %.6f\n", epoch, total_loss / train->num_samples);
    }

    /* Test evaluation */
    printf("\nTest predictions:\n");
    int correct = 0;
    for (int s = 0; s < test->num_samples; s++) {
        Tensor* h = cml_zeros(h_shape, 2, NULL);
        for (int t = 0; t < SEQ_LEN; t++) {
            float val = test_X[s * 4 + t];
            Tensor* x_t = cml_tensor(&val, x_shape, 2, NULL);
            h = gru_cell_forward(gru, x_t, h);
        }
        Tensor* pred = cml_nn_sequential_forward(head, h);
        float p = tensor_get_float(pred, 0);
        int cls = p > 0.5f ? 1 : 0;
        int true_cls = (test_y[s] > 0.5f) ? 1 : 0;
        correct += (cls == true_cls);
        if (s < 8)
            printf("  seq %d: pred=%.3f [class %d] true=%d %s\n",
                   s, p, cls, true_cls, cls == true_cls ? "OK" : "WRONG");
    }
    printf("Test accuracy: %d/%d (%.0f%%)\n", correct, test->num_samples,
           correct / (float)test->num_samples * 100);

    cml_cleanup();
    return 0;
}
