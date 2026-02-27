/**
 * Example 03: Logistic Regression (Binary Classification)
 *
 * Classify breast cancer tumors (malignant vs benign) using Linear + Sigmoid + BCE.
 * Dataset: Wisconsin Breast Cancer (569 samples, 30 features, 2 classes).
 */
#include "cml.h"
#include <stdio.h>

int main(void) {
    cml_init();
    printf("Example 03: Logistic Regression (Breast Cancer)\n\n");

    Dataset* ds = cml_dataset_load("breast_cancer");
    if (!ds) { printf("Failed to load breast_cancer dataset\n"); return 1; }

    dataset_normalize(ds, "minmax");
    Dataset *train, *test;
    dataset_split(ds, 0.8f, &train, &test);

    printf("Train: %d samples, Test: %d samples, Features: %d\n\n",
           train->num_samples, test->num_samples, train->input_size);

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(train->input_size, 1,
                          DTYPE_FLOAT32, DEVICE_CPU, true));
    cml_nn_sequential_add(model, (Module*)cml_nn_sigmoid());

    cml_summary((Module*)model);
    Optimizer* opt = cml_optim_adam_for_model((Module*)model, 0.05f, 0.0f, 0.9f, 0.999f, 1e-8f);

    for (int epoch = 1; epoch <= 50; epoch++) {
        Tensor* pred = cml_nn_sequential_forward(model, train->X);
        Tensor* loss = cml_nn_bce_loss(pred, train->y);
        cml_optim_zero_grad(opt);
        cml_backward(loss, NULL, false, false);
        cml_optim_step(opt);

        if (epoch % 10 == 0)
            printf("Epoch %3d  Loss: %.6f\n", epoch, tensor_get_float(loss, 0));
    }

    /* Evaluate on test set */
    Tensor* pred = cml_nn_sequential_forward(model, test->X);
    int correct = 0;
    for (int i = 0; i < test->num_samples; i++) {
        float p = tensor_get_float(pred, i);
        int cls = p > 0.5f ? 1 : 0;
        correct += (cls == (int)tensor_get_float(test->y, i));
    }
    printf("\nTest accuracy: %d/%d (%.1f%%)\n", correct, test->num_samples,
           correct / (float)test->num_samples * 100);

    printf("\nSample predictions:\n");
    for (int i = 0; i < 8 && i < test->num_samples; i++) {
        float p = tensor_get_float(pred, i);
        int cls = p > 0.5f ? 1 : 0;
        int true_cls = (int)tensor_get_float(test->y, i);
        printf("  pred=%.3f [%s] true=%s %s\n", p,
               cls ? "malignant" : "benign",
               true_cls ? "malignant" : "benign",
               cls == true_cls ? "OK" : "WRONG");
    }

    cml_cleanup();
    return 0;
}
