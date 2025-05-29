#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../include/Core/training.h"
#include "../../include/Core/autograd.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/logging.h"

// Helper function to initialize weights
static void initialize_weights(Node *weights, int input_size, int output_size)
{
    if (!weights || !weights->tensor)
        return;
    
    // Xavier initialization
    float scale = sqrtf(2.0f / (float)(input_size + output_size));
    
    for (int i = 0; i < input_size * output_size; i++) {
        weights->tensor->storage->data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

static void initialize_bias(Node *bias, int size)
{
    if (!bias || !bias->tensor)
        return;
    
    for (int i = 0; i < size; i++) {
        bias->tensor->storage->data[i] = 0.0f;
    }
}

// Create neural network
NeuralNetwork *create_neural_network(int input_size)
{
    NeuralNetwork *network = (NeuralNetwork *)cm_safe_malloc(sizeof(NeuralNetwork), __FILE__, __LINE__);
    if (!network)
        return NULL;
    
    network->layers = NULL;
    network->num_layers = 0;
    network->input_size = input_size;
    network->optimizer = NULL;
    network->loss_function = LOSS_MSE;
    network->is_training = 1;
    network->last_loss = 0.0f;
    network->parameters = NULL;
    network->num_parameters = 0;
    
    return network;
}

// Build network with optimizer and loss function
CM_Error build_network(NeuralNetwork *network, OptimizerType optimizer_type, 
                      float learning_rate, LossType loss_function, 
                      float l1_lambda, float l2_lambda)
{
    if (!network)
        return CM_NULL_POINTER_ERROR;
    
    // Initialize optimizer
    network->optimizer = (OptimizerState *)cm_safe_malloc(sizeof(OptimizerState), __FILE__, __LINE__);
    if (!network->optimizer)
        return CM_MEMORY_ALLOCATION_ERROR;
    
    network->optimizer->type = optimizer_type;
    network->optimizer->learning_rate = learning_rate;
    network->optimizer->l1_lambda = l1_lambda;
    network->optimizer->l2_lambda = l2_lambda;
    network->loss_function = loss_function;
    
    // Initialize optimizer-specific parameters
    switch (optimizer_type) {
        case OPTIMIZER_ADAM:
            network->optimizer->beta1 = 0.9f;
            network->optimizer->beta2 = 0.999f;
            network->optimizer->epsilon = 1e-8f;
            network->optimizer->t = 0;
            break;
        case OPTIMIZER_SGD:
            network->optimizer->momentum = 0.0f;
            break;
        case OPTIMIZER_RMSPROP:
            network->optimizer->alpha = 0.99f;
            network->optimizer->epsilon = 1e-8f;
            break;
    }
    
    return CM_SUCCESS;
}

// Add layer to network
CM_Error model_add(NeuralNetwork *network, LayerType type, ActivationType activation,
                  int input_size, int output_size, float rate, int kernel_size, int stride)
{
    if (!network)
        return CM_NULL_POINTER_ERROR;
    
    Layer *new_layer = (Layer *)cm_safe_malloc(sizeof(Layer), __FILE__, __LINE__);
    if (!new_layer)
        return CM_MEMORY_ALLOCATION_ERROR;
    
    new_layer->type = type;
    new_layer->activation = activation;
    new_layer->input_size = input_size;
    new_layer->output_size = output_size;
    new_layer->dropout_rate = rate;
    new_layer->kernel_size = kernel_size;
    new_layer->stride = stride;
    new_layer->next = NULL;
    new_layer->last_input = NULL;
    new_layer->last_output = NULL;
    new_layer->dropout_mask = NULL;
    
    // Initialize weights and bias for dense layers
    if (type == LAYER_DENSE) {
        int weight_sizes[2] = {input_size, output_size};
        new_layer->weights = empty(weight_sizes, 2);
        if (!new_layer->weights) {
            cm_safe_free((void **)&new_layer);
            return CM_MEMORY_ALLOCATION_ERROR;
        }
        
        int bias_sizes[1] = {output_size};
        new_layer->bias = empty(bias_sizes, 1);
        if (!new_layer->bias) {
            cm_safe_free((void **)&new_layer);
            return CM_MEMORY_ALLOCATION_ERROR;
        }
        
        // Initialize parameters
        initialize_weights(new_layer->weights, input_size, output_size);
        initialize_bias(new_layer->bias, output_size);
        
        // Set requires_grad for parameters
        set_requires_grad(new_layer->weights, 1);
        set_requires_grad(new_layer->bias, 1);
    }
    
    // Add layer to network
    if (network->layers == NULL) {
        network->layers = new_layer;
    } else {
        Layer *current = network->layers;
        while (current->next != NULL) {
            current = current->next;
        }
        current->next = new_layer;
    }
    
    network->num_layers++;
    return CM_SUCCESS;
}

// Apply activation function
static Node *apply_activation(Node *input, ActivationType activation)
{
    switch (activation) {
        case ACTIVATION_RELU:
            return relu(input);
        case ACTIVATION_SIGMOID:
            return sigmoid(input);
        case ACTIVATION_TANH:
            return tanh_tensor(input);
        case ACTIVATION_LINEAR:
        case ACTIVATION_NONE:
            return input;
        default:
            return input;
    }
}

// Forward pass through network
Node *forward_network(NeuralNetwork *network, Node *input)
{
    if (!network || !input)
        return NULL;
    
    Node *current_input = input;
    Layer *layer = network->layers;
    
    while (layer != NULL) {
        if (layer->type == LAYER_DENSE) {
            // For now, implement a simple linear transformation for scalar inputs
            // This is a simplified version - in a full implementation, you'd handle proper matrix operations
            
            // Compute weighted sum: sum(input_i * weight_ij) + bias_j for each output neuron
            Node *output = tensor(0.0f, 1);  // Start with zero
            
            // For simplified implementation, assume single input and single output for now
            // In practice, you'd iterate over all input-output connections
            Node *weighted = mul(current_input, layer->weights);
            output = add(weighted, layer->bias);
            
            // Apply activation
            output = apply_activation(output, layer->activation);
            if (!output)
                return NULL;
            
            // Store for backward pass
            layer->last_input = current_input;
            layer->last_output = output;
            
            current_input = output;
        }
        
        layer = layer->next;
    }
    
    return current_input;
}

// Backward pass through network
CM_Error backward_pass(NeuralNetwork *network, Node *loss)
{
    if (!network || !loss)
        return CM_NULL_POINTER_ERROR;
    
    // Compute gradients
    backward_from_root(loss);
    
    return CM_SUCCESS;
}

// Zero gradients for all parameters
CM_Error zero_grad_network(NeuralNetwork *network)
{
    if (!network)
        return CM_NULL_POINTER_ERROR;
    
    Layer *layer = network->layers;
    while (layer != NULL) {
        if (layer->weights) {
            layer->weights->grad = 0.0f;
            layer->weights->grad_accumulated = 0;
        }
        if (layer->bias) {
            layer->bias->grad = 0.0f;
            layer->bias->grad_accumulated = 0;
        }
        layer = layer->next;
    }
    
    return CM_SUCCESS;
}

// Optimizer step
CM_Error optimizer_step(NeuralNetwork *network)
{
    if (!network || !network->optimizer)
        return CM_NULL_POINTER_ERROR;
    
    float lr = network->optimizer->learning_rate;
    Layer *layer = network->layers;
    
    while (layer != NULL) {
        if (layer->type == LAYER_DENSE) {
            // Simple SGD update for now
            if (layer->weights && layer->weights->grad_accumulated) {
                for (int i = 0; i < layer->input_size * layer->output_size; i++) {
                    layer->weights->tensor->storage->data[i] -= lr * layer->weights->grad;
                }
            }
            
            if (layer->bias && layer->bias->grad_accumulated) {
                for (int i = 0; i < layer->output_size; i++) {
                    layer->bias->tensor->storage->data[i] -= lr * layer->bias->grad;
                }
            }
        }
        layer = layer->next;
    }
    
    return CM_SUCCESS;
}

// Training step
CM_Error train_step(NeuralNetwork *network, Node *input, Node *target)
{
    if (!network || !input || !target)
        return CM_NULL_POINTER_ERROR;
    
    // Zero gradients
    zero_grad_network(network);
    
    // Forward pass
    Node *output = forward_network(network, input);
    if (!output)
        return CM_COMPUTATION_ERROR;
    
    // Compute loss
    Node *loss = NULL;
    switch (network->loss_function) {
        case LOSS_MSE:
            loss = mse_loss(output, target, 0);  // Using autograd version
            break;
        default:
            loss = mse_loss(output, target, 0);  // Using autograd version
            break;
    }
    
    if (!loss)
        return CM_COMPUTATION_ERROR;
    
    network->last_loss = loss->tensor->storage->data[0];
    
    // Backward pass
    CM_Error error = backward_pass(network, loss);
    if (error != CM_SUCCESS)
        return error;
    
    // Optimizer step
    return optimizer_step(network);
}

// Train network
CM_Error train_network(NeuralNetwork *network, Dataset *dataset, int epochs)
{
    if (!network || !dataset)
        return CM_NULL_POINTER_ERROR;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;
        
        for (int i = 0; i < dataset->num_samples; i++) {
            // For simplified implementation, use first input element as scalar
            // In a full implementation, you'd handle multi-dimensional inputs properly
            Node *input = tensor(dataset->X[i][0], 0);  // Use first element
            Node *target = tensor(dataset->y[i][0], 0);  // Use first element
            
            if (!input || !target) {
                LOG_ERROR("Failed to create input/target tensors");
                continue;
            }
            
            CM_Error error = train_step(network, input, target);
            if (error != CM_SUCCESS) {
                LOG_ERROR("Training step failed: %d", error);
                continue;
            }
            
            total_loss += network->last_loss;
        }
        
        float avg_loss = total_loss / dataset->num_samples;
        LOG_INFO("Epoch %d/%d - Loss: %.6f", epoch + 1, epochs, avg_loss);
    }
    
    return CM_SUCCESS;
}

// Test network
CM_Error test_network(NeuralNetwork *network, float **X_test, float **y_test, 
                     int num_samples, float *accuracy)
{
    if (!network || !X_test || !y_test)
        return CM_NULL_POINTER_ERROR;
    
    float total_loss = 0.0f;
    int correct_predictions = 0;
    
    for (int i = 0; i < num_samples; i++) {
        // For simplified implementation, use first input element as scalar
        Node *input = tensor(X_test[i][0], 0);  // Use first element
        Node *target = tensor(y_test[i][0], 0);  // Use first element
        
        if (!input || !target)
            continue;
        
        Node *output = forward_network(network, input);
        if (!output)
            continue;
        
        Node *loss = mse_loss(output, target, 0);
        if (loss) {
            total_loss += loss->tensor->storage->data[0];
        }
        
        // Check prediction accuracy (for binary classification)
        float predicted = output->tensor->storage->data[0];
        float actual = target->tensor->storage->data[0];
        
        if ((predicted > 0.5f && actual > 0.5f) || (predicted <= 0.5f && actual <= 0.5f)) {
            correct_predictions++;
        }
        
        LOG_INFO("Test %d: Input=%.2f, Target=%.2f, Output=%.2f", 
                 i, X_test[i][0], actual, predicted);
    }
    
    float avg_loss = total_loss / num_samples;
    float acc = (float)correct_predictions / num_samples;
    
    LOG_INFO("Test Results - Loss: %.6f, Accuracy: %.2f%%", avg_loss, acc * 100.0f);
    
    if (accuracy) {
        *accuracy = acc;
    }
    
    return CM_SUCCESS;
}

// Print network summary
void summary(NeuralNetwork *network)
{
    if (!network) {
        LOG_INFO("Network is NULL");
        return;
    }
    
    LOG_INFO("Neural Network Summary:");
    LOG_INFO("Input size: %d", network->input_size);
    LOG_INFO("Number of layers: %d", network->num_layers);
    
    Layer *layer = network->layers;
    int layer_num = 1;
    
    while (layer != NULL) {
        const char *layer_type = (layer->type == LAYER_DENSE) ? "Dense" : "Unknown";
        const char *activation_name = "Unknown";
        
        switch (layer->activation) {
            case ACTIVATION_RELU: activation_name = "ReLU"; break;
            case ACTIVATION_SIGMOID: activation_name = "Sigmoid"; break;
            case ACTIVATION_TANH: activation_name = "Tanh"; break;
            case ACTIVATION_LINEAR: activation_name = "Linear"; break;
            case ACTIVATION_NONE: activation_name = "None"; break;
        }
        
        LOG_INFO("Layer %d: %s (%d -> %d) %s", 
                 layer_num, layer_type, layer->input_size, layer->output_size, activation_name);
        
        layer = layer->next;
        layer_num++;
    }
    
    if (network->optimizer) {
        const char *optimizer_name = "Unknown";
        switch (network->optimizer->type) {
            case OPTIMIZER_SGD: optimizer_name = "SGD"; break;
            case OPTIMIZER_ADAM: optimizer_name = "Adam"; break;
            case OPTIMIZER_RMSPROP: optimizer_name = "RMSprop"; break;
        }
        LOG_INFO("Optimizer: %s (lr=%.4f)", optimizer_name, network->optimizer->learning_rate);
    }
    
    const char *loss_name = (network->loss_function == LOSS_MSE) ? "MSE" : "Unknown";
    LOG_INFO("Loss function: %s", loss_name);
}

// Free neural network
void free_neural_network(NeuralNetwork *network)
{
    if (!network)
        return;
    
    Layer *layer = network->layers;
    while (layer != NULL) {
        Layer *next = layer->next;
        
        // Free layer parameters (weights and bias are autograd nodes, 
        // they should be freed by autograd system)
        
        cm_safe_free((void **)&layer);
        layer = next;
    }
    
    if (network->optimizer) {
        cm_safe_free((void **)&network->optimizer);
    }
    
    cm_safe_free((void **)&network);
}
