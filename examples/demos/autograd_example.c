#include "cml.h"
#include <stdio.h>
#include <stdlib.h>

// z = x^2 + y^2
static void example_simple_gradients(void) {
    printf("\n=== Example 1: Simple Gradients ===\n");
    printf("Computing gradients for: z = x^2 + y^2\n\n");

    int shape[]         = {1};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* x = tensor_ones(shape, 1, &config);
    Tensor* y = tensor_ones(shape, 1, &config);

    tensor_set_float(x, 0, 3.0f);
    tensor_set_float(y, 0, 4.0f);

    x->requires_grad = true;
    y->requires_grad = true;

    Tensor* exp = tensor_ones(shape, 1, &config);
    tensor_set_float(exp, 0, 2.0f);

    Tensor* x_squared = tensor_pow(x, exp);
    Tensor* y_squared = tensor_pow(y, exp);
    Tensor* z         = tensor_add(x_squared, y_squared);

    printf("Forward pass:\n");
    printf("  x = %.2f\n", (double)tensor_get_float(x, 0));
    printf("  y = %.2f\n", (double)tensor_get_float(y, 0));
    printf("  z = x^2 + y^2 = %.2f\n", (double)tensor_get_float(z, 0));

    tensor_backward(z, NULL, false, false);

    printf("\nBackward pass:\n");
    printf("  dz/dx = 2x = %.2f (expected: %.2f)\n", (double)tensor_get_float(x->grad, 0),
           (double)(2.0f * tensor_get_float(x, 0)));
    printf("  dz/dy = 2y = %.2f (expected: %.2f)\n", (double)tensor_get_float(y->grad, 0),
           (double)(2.0f * tensor_get_float(y, 0)));

    tensor_free(x);
    tensor_free(y);
    tensor_free(exp);
    tensor_free(x_squared);
    tensor_free(y_squared);
    tensor_free(z);
}

// y = sigmoid(w*x + b)
static void example_neural_network(void) {
    printf("\n=== Example 2: Neural Network ===\n");
    printf("Simple 1-layer network: y = sigmoid(w*x + b)\n\n");

    int shape[]         = {1};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* w = tensor_ones(shape, 1, &config);
    Tensor* b = tensor_ones(shape, 1, &config);
    Tensor* x = tensor_ones(shape, 1, &config);

    tensor_set_float(w, 0, 0.5f);
    tensor_set_float(b, 0, 0.1f);
    tensor_set_float(x, 0, 2.0f);

    w->requires_grad = true;
    b->requires_grad = true;

    Tensor* wx     = tensor_mul(w, x);
    Tensor* linear = tensor_add(wx, b);
    Tensor* y      = tensor_sigmoid(linear);

    printf("Forward pass:\n");
    printf("  w = %.2f\n", (double)tensor_get_float(w, 0));
    printf("  b = %.2f\n", (double)tensor_get_float(b, 0));
    printf("  x = %.2f\n", (double)tensor_get_float(x, 0));
    printf("  w*x = %.2f\n", (double)tensor_get_float(wx, 0));
    printf("  w*x + b = %.2f\n", (double)tensor_get_float(linear, 0));
    printf("  y = sigmoid(w*x + b) = %.4f\n", (double)tensor_get_float(y, 0));

    tensor_backward(y, NULL, false, false);

    printf("\nBackward pass (gradients):\n");
    printf("  dy/dw = %.4f\n", (double)tensor_get_float(w->grad, 0));
    printf("  dy/db = %.4f\n", (double)tensor_get_float(b->grad, 0));

    tensor_free(w);
    tensor_free(b);
    tensor_free(x);
    tensor_free(wx);
    tensor_free(linear);
    tensor_free(y);
}

static void example_loss_function(void) {
    printf("\n=== Example 3: Loss Function ===\n");
    printf("Training with MSE loss\n\n");

    int shape[]         = {3};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* prediction = tensor_empty(shape, 1, &config);
    Tensor* target     = tensor_empty(shape, 1, &config);

    tensor_set_float(prediction, 0, 1.0f);
    tensor_set_float(prediction, 1, 2.0f);
    tensor_set_float(prediction, 2, 3.0f);

    tensor_set_float(target, 0, 1.5f);
    tensor_set_float(target, 1, 2.5f);
    tensor_set_float(target, 2, 3.5f);

    prediction->requires_grad = true;

    Tensor* loss = tensor_mse_loss(prediction, target);

    printf("Forward pass:\n");
    printf("  Prediction: [%.1f, %.1f, %.1f]\n", (double)tensor_get_float(prediction, 0),
           (double)tensor_get_float(prediction, 1), (double)tensor_get_float(prediction, 2));
    printf("  Target:     [%.1f, %.1f, %.1f]\n", (double)tensor_get_float(target, 0),
           (double)tensor_get_float(target, 1), (double)tensor_get_float(target, 2));
    printf("  MSE Loss:   %.4f\n", (double)tensor_get_float(loss, 0));

    tensor_backward(loss, NULL, false, false);

    printf("\nBackward pass (gradients for prediction):\n");
    printf("  Gradient:   [%.4f, %.4f, %.4f]\n", (double)tensor_get_float(prediction->grad, 0),
           (double)tensor_get_float(prediction->grad, 1),
           (double)tensor_get_float(prediction->grad, 2));

    float learning_rate = 0.1f;
    printf("\nGradient descent step (lr=%.1f):\n", (double)learning_rate);
    for (size_t i = 0; i < prediction->numel; i++) {
        float old_val = tensor_get_float(prediction, i);
        float grad    = tensor_get_float(prediction->grad, i);
        float new_val = old_val - learning_rate * grad;
        printf("  pred[%zu]: %.4f -> %.4f (grad=%.4f)\n", i, (double)old_val, (double)new_val,
               (double)grad);
    }

    tensor_free(prediction);
    tensor_free(target);
    tensor_free(loss);
}

static void example_gradient_accumulation(void) {
    printf("\n=== Example 4: Gradient Accumulation ===\n");
    printf("Accumulating gradients from multiple losses\n\n");

    int shape[]         = {1};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* x = tensor_ones(shape, 1, &config);
    tensor_set_float(x, 0, 2.0f);
    x->requires_grad = true;

    Tensor* exp1 = tensor_ones(shape, 1, &config);
    tensor_set_float(exp1, 0, 2.0f);
    Tensor* y1 = tensor_pow(x, exp1);

    printf("First computation: y1 = x^2\n");
    printf("  x = %.2f\n", (double)tensor_get_float(x, 0));
    printf("  y1 = %.2f\n", (double)tensor_get_float(y1, 0));

    tensor_backward(y1, NULL, false, false);
    printf("  Gradient after y1.backward(): %.2f\n", (double)tensor_get_float(x->grad, 0));

    Tensor* exp2 = tensor_ones(shape, 1, &config);
    tensor_set_float(exp2, 0, 3.0f);
    Tensor* y2 = tensor_pow(x, exp2);

    printf("\nSecond computation: y2 = x^3\n");
    printf("  y2 = %.2f\n", (double)tensor_get_float(y2, 0));

    tensor_backward(y2, NULL, false, false);
    printf("  Gradient after y2.backward(): %.2f\n", (double)tensor_get_float(x->grad, 0));
    printf("  (Accumulated from both y1 and y2)\n");

    tensor_zero_grad(x);
    printf("\nAfter zero_grad():\n");
    printf("  Gradient: %s\n", x->grad ? "has value" : "NULL");

    tensor_free(x);
    tensor_free(exp1);
    tensor_free(exp2);
    tensor_free(y1);
    tensor_free(y2);
}

static void example_no_grad_mode(void) {
    printf("\n=== Example 5: No Gradient Mode ===\n");
    printf("Disabling gradient computation\n\n");

    int shape[]         = {1};
    TensorConfig config = (TensorConfig){
        .dtype = DTYPE_FLOAT32, .device = DEVICE_CPU, .has_dtype = true, .has_device = true};
    Tensor* x = tensor_ones(shape, 1, &config);
    tensor_set_float(x, 0, 3.0f);
    x->requires_grad = true;

    printf("With gradients enabled:\n");
    Tensor* exp1 = tensor_ones(shape, 1, &config);
    tensor_set_float(exp1, 0, 2.0f);
    Tensor* y1 = tensor_pow(x, exp1);
    printf("  y = x^2 = %.2f\n", (double)tensor_get_float(y1, 0));
    printf("  ir_node: %s\n", y1->ir_node ? "exists" : "NULL");

    autograd_no_grad_enter();

    printf("\nWith gradients disabled (no_grad mode):\n");
    Tensor* exp2 = tensor_ones(shape, 1, &config);
    tensor_set_float(exp2, 0, 2.0f);
    Tensor* y2 = tensor_pow(x, exp2);
    printf("  y = x^2 = %.2f\n", (double)tensor_get_float(y2, 0));
    printf("  ir_node: %s\n", y2->ir_node ? "exists" : "NULL");
    printf("  (No computation graph built!)\n");

    autograd_no_grad_exit();

    tensor_free(x);
    tensor_free(exp1);
    tensor_free(exp2);
    tensor_free(y1);
    tensor_free(y2);
}

int main(void) {
    cml_init();

    example_simple_gradients();
    example_neural_network();
    example_loss_function();
    example_gradient_accumulation();
    example_no_grad_mode();
    cml_cleanup();

    return 0;
}
