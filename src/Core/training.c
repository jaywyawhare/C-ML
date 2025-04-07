#include "../../include/Core/training.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"
#include "../../include/Layers/dense.h"
#include "../../include/Layers/flatten.h"
#include "../../include/Layers/dropout.h"
#include "../../include/Layers/maxpooling.h"
#include "../../include/Layers/pooling.h"
#include "../../include/Activations/relu.h"
#include "../../include/Activations/sigmoid.h"
#include "../../include/Activations/tanh.h"
#include "../../include/Activations/softmax.h"
#include "../../include/Activations/leaky_relu.h"
#include "../../include/Activations/elu.h"
#include "../../include/Activations/gelu.h"
#include "../../include/Activations/linear.h"
#include "../../include/Loss_Functions/binary_cross_entropy_loss.h"
#include "../../include/Loss_Functions/cosine_similarity_loss.h"
#include "../../include/Loss_Functions/focal_loss.h"
#include "../../include/Loss_Functions/huber_loss.h"
#include "../../include/Loss_Functions/kld_loss.h"
#include "../../include/Loss_Functions/log_cosh_loss.h"
#include "../../include/Loss_Functions/mean_squared_error.h"
#include "../../include/Loss_Functions/poisson_loss.h"
#include "../../include/Loss_Functions/smooth_l1_loss.h"
#include "../../include/Loss_Functions/tversky_loss.h"
#include "../../include/Metrics/accuracy.h"
#include "../../include/Metrics/balanced_accuracy.h"
#include "../../include/Metrics/cohens_kappa.h"
#include "../../include/Metrics/f1_score.h"
#include "../../include/Metrics/iou.h"
#include "../../include/Metrics/mcc.h"
#include "../../include/Metrics/mean_absolute_error.h"
#include "../../include/Metrics/mean_absolute_percentage_error.h"
#include "../../include/Metrics/precision.h"
#include "../../include/Metrics/recall.h"
#include "../../include/Metrics/reduce_mean.h"
#include "../../include/Metrics/root_mean_squared_error.h"
#include "../../include/Metrics/r2_score.h"
#include "../../include/Metrics/specificity.h"
#include "../../include/Regularizers/l1.h"
#include "../../include/Regularizers/l2.h"
#include "../../include/Regularizers/l1_l2.h"
#include "../../include/Optimizers/adam.h"
#include "../../include/Optimizers/rmsprop.h"
#include "../../include/Optimizers/sgd.h"
#include "../../include/Core/dataset.h" 

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>

/**
 * @brief Create a new neural network
 *
 * @param input_size The size of the input layer
 * @return NeuralNetwork* Pointer to the new neural network
 */
NeuralNetwork *create_neural_network(int input_size)
{
    NeuralNetwork *network = (NeuralNetwork *)cm_safe_malloc(sizeof(NeuralNetwork), __FILE__, __LINE__);
    if (network == NULL)
    {
        return NULL;
    }
    network->head = NULL;
    network->tail = NULL;
    network->num_layers = 0;
    network->input_size = input_size;
    network->max_layer_output_size = input_size; /* Initialize with input_size as the default */
    network->optimizer_type = OPTIMIZER_NONE;
    network->loss_function = LOSS_MSE;
    network->learning_rate = 0.01f;
    network->l1_lambda = 0.0f;
    network->l2_lambda = 0.0f;
    network->beta1 = 0.9f;
    network->beta2 = 0.999f;
    network->epsilon = 1e-7f;
    network->cache_w = NULL;
    network->cache_b = NULL;
    network->v_w = NULL;
    network->v_b = NULL;
    network->s_w = NULL;
    network->s_b = NULL;
    return network;
}

/**
 * @brief Build the neural network by setting the optimizer, loss function, and regularization parameters.
 *
 * @param network Pointer to the neural network.
 * @param optimizer_type The type of optimization algorithm to use (SGD, RMSProp, Adam).
 * @param learning_rate The learning rate for the optimizer.
 * @param loss_function The loss function to use.
 * @param l1_lambda L1 regularization parameter
 * @param l2_lambda L2 regularization parameter
 * @return CM_Error Error code.
 */
CM_Error build_network(NeuralNetwork *network, OptimizerType optimizer_type, float learning_rate, int loss_function, float l1_lambda, float l2_lambda)
{
    if (network == NULL)
    {
        LOG_ERROR("Null pointer argument");
        return CM_NULL_POINTER_ERROR;
    }

    network->optimizer_type = optimizer_type;
    network->loss_function = loss_function;
    network->learning_rate = learning_rate;
    network->l1_lambda = l1_lambda;
    network->l2_lambda = l2_lambda;

    /* 
     * Note: We don't allocate optimizer parameters here.
     * They will be allocated in initialize_optimizer_params() which is called at the start of training.
     * This ensures we allocate the correct amount of memory based on the final network structure.
     */
    
    return CM_SUCCESS;
}

/**
 * @brief Calculate the maximum input and output sizes needed for the network.
 * 
 * This function traverses the network to find the maximum input and output sizes
 * across all layers, which can be used to allocate memory safely.
 *
 * @param network Pointer to the neural network.
 * @param max_input_size Pointer to store the maximum input size.
 * @param max_output_size Pointer to store the maximum output size.
 * @return CM_Error Error code.
 */
CM_Error calculate_max_buffer_sizes(NeuralNetwork *network, int *max_input_size, int *max_output_size)
{
    if (network == NULL || max_input_size == NULL || max_output_size == NULL)
    {
        LOG_ERROR("Null pointer argument");
        return CM_NULL_POINTER_ERROR;
    }

    *max_input_size = network->input_size;
    *max_output_size = 0;

    NeuralNetworkNode *current = network->head;
    while (current != NULL)
    {
        switch (current->type)
        {
        case LAYER_DENSE:
        {
            DenseLayer *dense = (DenseLayer *)current->layer;
            if (dense->input_size > *max_input_size)
            {
                *max_input_size = dense->input_size;
            }
            if (dense->output_size > *max_output_size)
            {
                *max_output_size = dense->output_size;
            }
            break;
        }
        case LAYER_FLATTEN:
        {
            FlattenLayer *flatten = (FlattenLayer *)current->layer;
            if (flatten->output_size > *max_output_size)
            {
                *max_output_size = flatten->output_size;
            }
            break;
        }
        case LAYER_MAXPOOLING:
        {
            MaxPoolingLayer *mp = (MaxPoolingLayer *)current->layer;
            int out_size = compute_maxpooling_output_size(network->input_size, mp->kernel_size, mp->stride);
            if (out_size > *max_output_size)
            {
                *max_output_size = out_size;
            }
            break;
        }
        case LAYER_POOLING:
        {
            PollingLayer *pooling = (PollingLayer *)current->layer;
            int out_size = compute_polling_output_size(network->input_size, pooling->kernel_size, pooling->stride);
            if (out_size > *max_output_size)
            {
                *max_output_size = out_size;
            }
            break;
        }
        default:
            break;
        }
        current = current->next;
    }

    /* Ensure max_output_size is at least as large as the network's output size */
    if (*max_output_size < network->max_layer_output_size)
    {
        *max_output_size = network->max_layer_output_size;
    }

    LOG_DEBUG("Calculated max buffer sizes: input_size=%d, output_size=%d", *max_input_size, *max_output_size);
    return CM_SUCCESS;
}

/**
 * @brief Initialize optimizer parameters based on the current network structure.
 * 
 * This function allocates memory for optimizer parameters based on the maximum layer size.
 * It should be called before training starts, after all layers have been added.
 *
 * @param network Pointer to the neural network.
 * @return CM_Error Error code.
 */
CM_Error initialize_optimizer_params(NeuralNetwork *network)
{
    if (network == NULL)
    {
        LOG_ERROR("Network is NULL");
        return CM_NULL_POINTER_ERROR;
    }

    /* Free any previously allocated optimizer parameters */
    cm_safe_free((void **)&network->v_w);
    cm_safe_free((void **)&network->v_b);
    cm_safe_free((void **)&network->s_w);
    cm_safe_free((void **)&network->s_b);
    cm_safe_free((void **)&network->cache_w);
    cm_safe_free((void **)&network->cache_b);

    /* Calculate the maximum parameter size based on the largest layer */
    int max_param_size = network->max_layer_output_size * network->max_layer_output_size;
    
    LOG_DEBUG("Initializing optimizer parameters with max_layer_output_size: %d", network->max_layer_output_size);
    
    if (network->optimizer_type == OPTIMIZER_ADAM)
    {
        network->v_w = (float *)cm_safe_malloc(max_param_size * sizeof(float), __FILE__, __LINE__);
        network->v_b = (float *)cm_safe_malloc(network->max_layer_output_size * sizeof(float), __FILE__, __LINE__);
        network->s_w = (float *)cm_safe_malloc(max_param_size * sizeof(float), __FILE__, __LINE__);
        network->s_b = (float *)cm_safe_malloc(network->max_layer_output_size * sizeof(float), __FILE__, __LINE__);

        if (!network->v_w || !network->v_b || !network->s_w || !network->s_b)
        {
            LOG_ERROR("Memory allocation failed for Adam optimizer parameters");
            cm_safe_free((void **)&network->v_w);
            cm_safe_free((void **)&network->v_b);
            cm_safe_free((void **)&network->s_w);
            cm_safe_free((void **)&network->s_b);
            return CM_MEMORY_ALLOCATION_ERROR;
        }

        memset(network->v_w, 0, max_param_size * sizeof(float));
        memset(network->v_b, 0, network->max_layer_output_size * sizeof(float));
        memset(network->s_w, 0, max_param_size * sizeof(float));
        memset(network->s_b, 0, network->max_layer_output_size * sizeof(float));
    }
    else if (network->optimizer_type == OPTIMIZER_RMSPROP)
    {
        network->cache_w = (float *)cm_safe_malloc(max_param_size * sizeof(float), __FILE__, __LINE__);
        network->cache_b = (float *)cm_safe_malloc(network->max_layer_output_size * sizeof(float), __FILE__, __LINE__);

        if (!network->cache_w || !network->cache_b)
        {
            LOG_ERROR("Memory allocation failed for RMSProp optimizer parameters");
            cm_safe_free((void **)&network->cache_w);
            cm_safe_free((void **)&network->cache_b);
            return CM_MEMORY_ALLOCATION_ERROR;
        }

        memset(network->cache_w, 0, max_param_size * sizeof(float));
        memset(network->cache_b, 0, network->max_layer_output_size * sizeof(float));
    }

    return CM_SUCCESS;
}

/**
 * @brief Apply activation function to a single value
 *
 * @param value Input value
 * @param activation Activation type
 * @return float Activated value
 */
float apply_activation(float value, ActivationType activation)
{
    switch (activation)
    {
    case ACTIVATION_RELU:
        return relu(value);
    case ACTIVATION_SIGMOID:
        return sigmoid(value);
    case ACTIVATION_TANH:
        return tanH(value);
    case ACTIVATION_LEAKY_RELU:
        return leaky_relu(value);
    case ACTIVATION_ELU:
        return elu(value, 1.0);
    case ACTIVATION_GELU:
        return gelu(value);
    case ACTIVATION_LINEAR:
        return linear(value);
    case ACTIVATION_NONE:
    default:
        return value;
    }
}

/**
 * @brief Apply activation function to an array of values
 *
 * @param values Input array
 * @param size Size of the array
 * @param activation Activation type
 */
void apply_activation_array(float *values, int size, ActivationType activation)
{
    if (activation == ACTIVATION_SOFTMAX)
    {
        float *softmax_output = softmax(values, size);
        if (softmax_output == (float *)CM_NULL_POINTER_ERROR || softmax_output == (float *)CM_MEMORY_ALLOCATION_ERROR || softmax_output == (float *)CM_DIVISION_BY_ZERO_ERROR)
        {
            LOG_ERROR("Softmax activation failed");
            return;
        }
        for (int i = 0; i < size; i++)
        {
            values[i] = softmax_output[i];
        }
        cm_safe_free((void **)&softmax_output);
        return;
    }

    for (int i = 0; i < size; i++)
    {
        values[i] = apply_activation(values[i], activation);
    }
}

/**
 * @brief Add a layer to the neural network
 *
 * @param network Pointer to the neural network
 * @param config Layer configuration
 * @return CM_Error Error code
 */
CM_Error add_layer(NeuralNetwork *network, LayerConfig config)
{
    if (network == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    NeuralNetworkNode *new_node = (NeuralNetworkNode *)cm_safe_malloc(sizeof(NeuralNetworkNode), __FILE__, __LINE__);
    if (new_node == NULL)
    {
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    new_node->type = config.type;
    new_node->activation = config.activation;
    new_node->next = NULL;

    CM_Error error = CM_SUCCESS;

    switch (config.type)
    {
    case LAYER_DENSE:
    {
        DenseLayer *dense = (DenseLayer *)cm_safe_malloc(sizeof(DenseLayer), __FILE__, __LINE__);
        if (dense == NULL)
        {
            cm_safe_free((void **)&new_node);
            return CM_MEMORY_ALLOCATION_ERROR;
        }
        error = initialize_dense(dense, config.params.dense.input_size, config.params.dense.output_size);
        if (error != CM_SUCCESS)
        {
            cm_safe_free((void **)&dense);
            cm_safe_free((void **)&new_node);
            return error;
        }
        new_node->layer = dense;
        break;
    }
    case LAYER_FLATTEN:
    {
        FlattenLayer *flatten = (FlattenLayer *)cm_safe_malloc(sizeof(FlattenLayer), __FILE__, __LINE__);
        if (flatten == NULL)
        {
            cm_safe_free((void **)&new_node);
            return CM_MEMORY_ALLOCATION_ERROR;
        }
        error = initialize_flatten(flatten, config.params.dense.input_size);
        if (error != CM_SUCCESS)
        {
            cm_safe_free((void **)&flatten);
            cm_safe_free((void **)&new_node);
            return error;
        }
        new_node->layer = flatten;
        break;
    }
    case LAYER_DROPOUT:
    {
        DropoutLayer *dropout = (DropoutLayer *)cm_safe_malloc(sizeof(DropoutLayer), __FILE__, __LINE__);
        if (dropout == NULL)
        {
            cm_safe_free((void **)&new_node);
            return CM_MEMORY_ALLOCATION_ERROR;
        }
        error = initialize_dropout(dropout, config.params.dropout.rate);
        if (error != CM_SUCCESS)
        {
            cm_safe_free((void **)&dropout);
            cm_safe_free((void **)&new_node);
            return error;
        }
        new_node->layer = dropout;
        break;
    }
    case LAYER_MAXPOOLING:
    {
        MaxPoolingLayer *maxpooling = (MaxPoolingLayer *)cm_safe_malloc(sizeof(MaxPoolingLayer), __FILE__, __LINE__);
        if (maxpooling == NULL)
        {
            cm_safe_free((void **)&new_node);
            return CM_MEMORY_ALLOCATION_ERROR;
        }
        error = initialize_maxpooling(maxpooling, config.params.pooling.kernel_size, config.params.pooling.stride);
        if (error != CM_SUCCESS)
        {
            cm_safe_free((void **)&maxpooling);
            cm_safe_free((void **)&new_node);
            return error;
        }
        new_node->layer = maxpooling;
        break;
    }
    case LAYER_POOLING:
        cm_safe_free((void **)&new_node);
        return CM_INVALID_PARAMETER_ERROR;
    default:
        cm_safe_free((void **)&new_node);
        return CM_INVALID_PARAMETER_ERROR;
    }

    if (network->head == NULL)
    {
        network->head = new_node;
        network->tail = new_node;
    }
    else
    {
        network->tail->next = new_node;
        network->tail = new_node;
    }

    network->num_layers++;
    return CM_SUCCESS;
}

/**
 * @brief Adds a layer to the neural network using a simplified interface.
 *
 * This function simplifies the process of adding layers to the neural network
 * by taking layer parameters directly and constructing the LayerConfig internally.
 *
 * @param network Pointer to the neural network.
 * @param type The type of layer to add (e.g., LAYER_DENSE, LAYER_FLATTEN).
 * @param activation The activation function to use (e.g., ACTIVATION_RELU).
 * @param input_size The input size for dense layers.
 * @param output_size The output size for dense layers.
 * @param rate The dropout rate for dropout layers.
 * @param kernel_size The kernel size for pooling layers.
 * @param stride The stride for pooling layers.
 * @return CM_Error Error code.
 */
CM_Error model_add(NeuralNetwork *network, LayerType type, ActivationType activation,
                   int input_size, int output_size, float rate, int kernel_size, int stride)
{
    if (network == NULL)
    {
        LOG_ERROR("Network is NULL");
        return CM_NULL_POINTER_ERROR;
    }

    LayerConfig config;
    config.type = type;
    config.activation = activation;

    switch (type)
    {
    case LAYER_DENSE:
        config.params.dense.input_size = input_size;
        config.params.dense.output_size = output_size;
        
        /* Update max_layer_output_size if this layer's output is larger */
        if (output_size > network->max_layer_output_size)
        {
            network->max_layer_output_size = output_size;
            LOG_INFO("Updated network->max_layer_output_size to %d", network->max_layer_output_size);
        }
        break;
    case LAYER_DROPOUT:
        config.params.dropout.rate = rate;
        break;
    case LAYER_MAXPOOLING:
    case LAYER_POOLING:
        config.params.pooling.kernel_size = kernel_size;
        config.params.pooling.stride = stride;
        break;
    default:
        LOG_ERROR("Unsupported layer type: %d", type);
        return CM_INVALID_PARAMETER_ERROR;
    }

    return add_layer(network, config);
}

/**
 * @brief Perform forward pass through the network
 *
 * @param network Pointer to the neural network
 * @param input Input data
 * @param output Output data
 * @param input_size Size of the input
 * @param output_size Size of the output
 * @param is_training Whether in training mode (affects dropout)
 * @return CM_Error Error code
 */
CM_Error forward_pass(NeuralNetwork *network, float *input, float *output, int input_size, int output_size, int is_training)
{
    if (network == NULL || input == NULL || output == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (network->head == NULL)
    {
        return CM_LAYER_NOT_INITIALIZED_ERROR;
    }

    /* Calculate the maximum buffer sizes needed for this network */
    int max_input_size, max_output_size;
    CM_Error size_error = calculate_max_buffer_sizes(network, &max_input_size, &max_output_size);
    if (size_error != CM_SUCCESS)
    {
        return size_error;
    }

    /* Ensure max_input_size is at least as large as the provided input_size */
    if (max_input_size < input_size)
    {
        max_input_size = input_size;
    }

    /* Ensure max_output_size is at least as large as the provided output_size */
    if (max_output_size < output_size)
    {
        max_output_size = output_size;
    }

    /* Allocate memory for layer_input */
    float *layer_input = (float *)cm_safe_malloc(max_input_size * sizeof(float), __FILE__, __LINE__);
    if (layer_input == NULL)
    {
        LOG_ERROR("Unable to allocate memory for layer_input");
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    /* Allocate memory for layer_output */
    LOG_DEBUG("Allocating memory for layer_output. Input Size: %d, Output Size: %d, Max Input Size: %d, Max Output Size: %d", 
              input_size, output_size, max_input_size, max_output_size);
    float *layer_output = (float *)cm_safe_malloc(max_output_size * sizeof(float), __FILE__, __LINE__);
    if (layer_output == NULL)
    {
        LOG_ERROR("Unable to allocate memory for layer_output");
        cm_safe_free((void **)&layer_input);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    memcpy(layer_input, input, input_size * sizeof(float));

    NeuralNetworkNode *current = network->head;
    CM_Error error = CM_SUCCESS;
    int current_size = input_size;

    while (current != NULL)
    {
        switch (current->type)
        {
        case LAYER_DENSE:
        {
            DenseLayer *dense = (DenseLayer *)current->layer;
            if (dense->input_size != current_size)
            {
                LOG_ERROR("Input Size mismatch - current: %d layer input size: %d", current_size, dense->input_size);
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return CM_INVALID_LAYER_DIMENSIONS_ERROR;
            }

            error = forward_dense(dense, layer_input, layer_output);
            if (error != CM_SUCCESS)
            {
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return error;
            }

            apply_activation_array(layer_output, dense->output_size, current->activation);
            current_size = dense->output_size;
            break;
        }
        case LAYER_FLATTEN:
        {
            FlattenLayer *flatten = (FlattenLayer *)current->layer;
            error = forward_flatten(flatten, layer_input, layer_output);
            if (error != CM_SUCCESS)
            {
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return error;
            }
            current_size = flatten->output_size;
            break;
        }
        case LAYER_DROPOUT:
        {
            if (is_training)
            {
                DropoutLayer *dropout = (DropoutLayer *)current->layer;
                error = forward_dropout(dropout, layer_input, layer_output, current_size);
            }
            else
            {
                memcpy(layer_output, layer_input, current_size * sizeof(float));
            }
            if (error != CM_SUCCESS)
            {
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return error;
            }
            break;
        }
        case LAYER_MAXPOOLING:
        {
            MaxPoolingLayer *maxpooling = (MaxPoolingLayer *)current->layer;
            error = forward_maxpooling(maxpooling, layer_input, layer_output, current_size);
            if (error != CM_SUCCESS)
            {
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return error;
            }
            current_size = compute_maxpooling_output_size(current_size, maxpooling->kernel_size, maxpooling->stride);
            break;
        }
        case LAYER_POOLING:
        {
            PollingLayer *pooling = (PollingLayer *)current->layer;
            int new_size = compute_polling_output_size(current_size, pooling->kernel_size, pooling->stride);
            if (new_size < 0)
            {
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return CM_INVALID_PARAMETER_ERROR;
            }
            error = forward_polling(pooling, layer_input, layer_output, current_size);
            if (error != CM_SUCCESS)
            {
                cm_safe_free((void **)&layer_input);
                cm_safe_free((void **)&layer_output);
                return error;
            }
            current_size = new_size;
            break;
        }
        default:
            cm_safe_free((void **)&layer_input);
            cm_safe_free((void **)&layer_output);
            return CM_INVALID_PARAMETER_ERROR;
        }

        float *temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;

        current = current->next;
    }

    memcpy(output, layer_input, output_size * sizeof(float));

    cm_safe_free((void **)&layer_input);
    cm_safe_free((void **)&layer_output);

    return CM_SUCCESS;
}

/**
 * @brief Calculate the loss between predicted and actual values
 *
 * @param predicted Pointer to predicted values
 * @param actual Pointer to actual values
 * @param size Number of values
 * @param loss_type Type of loss function to use
 * @return float Calculated loss
 */
float calculate_loss(float *predicted, float *actual, int size, LossType loss_type)
{
    switch (loss_type)
    {
    case LOSS_MSE:
        return mean_squared_error(predicted, actual, size);
    case LOSS_BINARY_CROSS_ENTROPY:
        return binary_cross_entropy_loss(actual, predicted, size);
    case LOSS_FOCAL:
        return focal_loss(predicted, actual, size, 2.0);
    case LOSS_HUBER:
        return huber_loss(predicted, actual, size);
    case LOSS_KLD:
        return kld_loss(predicted, actual, size);
    case LOSS_LOG_COSH:
        return log_cosh_loss(predicted, actual, size);
    case LOSS_POISSON:
        return poisson_loss(predicted, actual, size);
    case LOSS_SMOOTH_L1:
        return smooth_l1_loss(predicted, actual, size);
    case LOSS_TVERSKY:
        return tversky_loss(predicted, actual, size);
    case LOSS_COSINE_SIMILARITY:
        return cosine_similarity_loss(predicted, actual, size);
    default:
        LOG_ERROR("Unknown loss type: %d", loss_type);
        return -1.0f;
    }
}

/**
 * @brief Calculate the gradient of the loss function
 *
 * @param predicted Pointer to predicted values
 * @param actual Pointer to actual values
 * @param gradient Pointer to store the calculated gradient
 * @param size Number of values
 * @param loss_type Type of loss function
 */
void calculate_loss_gradient(float *predicted, float *actual, float *gradient, int size, LossType loss_type)
{
    int i;
    switch (loss_type)
    {
    case LOSS_MSE:
        for (i = 0; i < size; i++)
        {
            gradient[i] = mean_squared_error_derivative(predicted[i], actual[i], size);
        }
        break;
    case LOSS_BINARY_CROSS_ENTROPY:
        for (i = 0; i < size; i++)
        {
            gradient[i] = binary_cross_entropy_loss_derivative(actual[i], predicted[i]);
        }
        break;
    case LOSS_FOCAL:
        for (i = 0; i < size; i++)
        {
            gradient[i] = focal_loss_derivative(actual[i], predicted[i], 2.0);
        }
        break;
    case LOSS_HUBER:
        for (i = 0; i < size; i++)
        {
            gradient[i] = huber_loss_derivative(predicted[i], actual[i]);
        }
        break;
    case LOSS_KLD:
        for (i = 0; i < size; i++)
        {
            gradient[i] = kld_loss_derivative(predicted[i], actual[i]);
        }
        break;
    case LOSS_LOG_COSH:
        for (i = 0; i < size; i++)
        {
            gradient[i] = log_cosh_loss_derivative(predicted[i], actual[i]);
        }
        break;
    case LOSS_POISSON:
        for (i = 0; i < size; i++)
        {
            gradient[i] = poisson_loss_derivative(predicted[i], actual[i]);
        }
        break;
    case LOSS_SMOOTH_L1:
        for (i = 0; i < size; i++)
        {
            gradient[i] = smooth_l1_loss_derivative(predicted[i], actual[i]);
        }
        break;
    case LOSS_TVERSKY:
    {
        float dt = tversky_loss_derivative(predicted, actual, size);
        for (i = 0; i < size; i++)
        {
            gradient[i] = dt;
        }
        break;
    }
    case LOSS_COSINE_SIMILARITY:
    {
        float dc = cosine_similarity_loss_derivative(predicted, actual, size);
        for (i = 0; i < size; i++)
        {
            gradient[i] = dc;
        }
        break;
    }
    default:
        LOG_ERROR("Unknown loss type: %d", loss_type);
        break;
    }
}

/**
 * @brief Train the neural network
 *
 * @param network Pointer to the neural network
 * @param dataset Pointer to the dataset
 * @param epochs Number of training epochs
 * @return CM_Error Error code
 */
CM_Error train_network(NeuralNetwork *network, Dataset *dataset, int epochs, ...)
{
    if (network == NULL || dataset == NULL)
    {
        LOG_ERROR("Null pointer argument.");
        return CM_NULL_POINTER_ERROR;
    }

    if (network->head == NULL)
    {
        LOG_ERROR("Neural network has no layers.");
        return CM_LAYER_NOT_INITIALIZED_ERROR;
    }

    if (network->optimizer_type == OPTIMIZER_NONE)
    {
        LOG_ERROR("Optimizer is not set. Call build_network first");
        return CM_OPTIMIZER_NOT_INITIALIZED_ERROR;
    }

    int num_samples = dataset->num_samples;
    int input_size = dataset->input_dim;
    int output_size = dataset->output_dim;

    if (num_samples <= 0 || input_size <= 0 || output_size <= 0)
    {
        LOG_ERROR("Invalid dataset dimensions.");
        return CM_INVALID_PARAMETER_ERROR;
    }

    // Set default batch size
    int batch_size = 32;

    // Parse optional batch size argument
    va_list args;
    va_start(args, epochs);
    if (epochs > 0) // Ensure optional arguments exist
    {
        int provided_batch_size = va_arg(args, int);
        if (provided_batch_size > 0) // Validate the provided batch size
        {
            batch_size = provided_batch_size;
        }
    }
    va_end(args);
    
    /* Initialize optimizer parameters based on the final network structure */
    CM_Error error = initialize_optimizer_params(network);
    if (error != CM_SUCCESS)
    {
        LOG_ERROR("Failed to initialize optimizer parameters");
        return error;
    }

    /* Calculate the maximum buffer sizes needed for this network */
    int max_input_size, max_output_size;
    CM_Error size_error = calculate_max_buffer_sizes(network, &max_input_size, &max_output_size);
    if (size_error != CM_SUCCESS)
    {
        return size_error;
    }

    /* Ensure max_output_size is at least as large as the provided output_size */
    if (max_output_size < output_size)
    {
        max_output_size = output_size;
    }

    /* Allocate memory for predictions */
    float *predictions = (float *)cm_safe_malloc(max_output_size * sizeof(float), __FILE__, __LINE__);
    if (predictions == NULL)
    {
        LOG_ERROR("Memory allocation failed for predictions.");
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    /* Allocate memory for loss_gradient */
    float *loss_gradient = (float *)cm_safe_malloc(max_output_size * sizeof(float), __FILE__, __LINE__);
    if (loss_gradient == NULL)
    {
        LOG_ERROR("Memory allocation failed for loss_gradient.");
        cm_safe_free((void **)&predictions);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0.0f;

        for (int sample = 0; sample < num_samples; sample += batch_size)
        {
            int current_batch_size = (sample + batch_size > num_samples) ? (num_samples - sample) : batch_size;

            for (int b = 0; b < current_batch_size; b++)
            {
                int idx = sample + b;

                if (dataset->X[idx] == NULL || dataset->y[idx] == NULL)
                {
                    LOG_ERROR("Null pointer in dataset at index %d.", idx);
                    cm_safe_free((void **)&predictions);
                    cm_safe_free((void **)&loss_gradient);
                    return CM_NULL_POINTER_ERROR;
                }

                CM_Error error = forward_pass(network, dataset->X[idx], predictions, input_size, output_size, 1);
                if (error != CM_SUCCESS)
                {
                    LOG_ERROR("Forward pass failed at index %d.", idx);
                    cm_safe_free((void **)&predictions);
                    cm_safe_free((void **)&loss_gradient);
                    return error;
                }

                float loss = calculate_loss(predictions, dataset->y[idx], output_size, network->loss_function);
                total_loss += loss;

                calculate_loss_gradient(predictions, dataset->y[idx], loss_gradient, output_size, network->loss_function);

                NeuralNetworkNode *current = network->head;

                while (current != NULL)
                {
                    if (current->type == LAYER_DENSE)
                    {
                        DenseLayer *dense = (DenseLayer *)current->layer;

                        if (dense == NULL || dense->weights == NULL || dense->biases == NULL)
                        {
                            LOG_ERROR("Null pointer in DenseLayer.");
                            cm_safe_free((void **)&predictions);
                            cm_safe_free((void **)&loss_gradient);
                            return CM_NULL_POINTER_ERROR;
                        }

                        if (network->l1_lambda > 0.0f || network->l2_lambda > 0.0f)
                        {
                            l1_l2(dense->weights, loss_gradient, network->l1_lambda, network->l2_lambda, dense->input_size * dense->output_size);
                        }

                        for (int i = 0; i < dense->output_size; i++)
                        {
                            for (int j = 0; j < dense->input_size; j++)
                            {
                                /* Only access dataset->X[idx][j] if j is within the bounds of input_size */
                                float x = (j < input_size) ? dataset->X[idx][j]  : 0.0f;
                                float *w = &dense->weights[i * dense->input_size + j];
                                float *b = &dense->biases[i];
                                float lr = network->learning_rate;

                                switch (network->optimizer_type)
                                {
                                case OPTIMIZER_SGD:
                                    update_sgd(w, b, loss_gradient[i], x, lr);
                                    break;
                                case OPTIMIZER_RMSPROP:
                                    update_rmsprop(w, b,
                                                   &dense->rmsprop_cache_w[i * dense->input_size + j],
                                                   &dense->rmsprop_cache_b[i],
                                                   loss_gradient[i], x, lr,
                                                   0.9f, network->epsilon);
                                    break;
                                case OPTIMIZER_ADAM:
                                    update_adam(w, b,
                                                &dense->adam_v_w[i * dense->input_size + j],
                                                &dense->adam_v_b[i],
                                                &dense->adam_s_w[i * dense->input_size + j],
                                                &dense->adam_s_b[i],
                                                loss_gradient[i], x, lr,
                                                network->beta1, network->beta2,
                                                network->epsilon, epoch);
                                    break;
                                default:
                                    LOG_ERROR("Unknown optimizer type: %d", network->optimizer_type);
                                    cm_safe_free((void **)&predictions);
                                    cm_safe_free((void **)&loss_gradient);
                                    return CM_INVALID_PARAMETER_ERROR;
                                }
                            }
                        }
                    }
                    current = current->next;
                }
            }
        }
       
        LOG_INFO("Epoch %d/%d - Loss: %.4f", epoch + 1, epochs, total_loss / num_samples);
    }

    cm_safe_free((void **)&predictions);
    cm_safe_free((void **)&loss_gradient);

    return CM_SUCCESS;
}

/**
 * @brief Evaluate the neural network on a given dataset.
 *
 * @param network Pointer to the neural network.
 * @param X_test Test data.
 * @param y_test Test labels.
 * @param num_samples Number of test samples.
 * @param input_size Size of each input sample.
 * @param output_size Size of each output sample.
 * @param metrics Array of evaluation metrics to calculate.
 * @param num_metrics Number of evaluation metrics to calculate.
 * @param results Array to store the results of the evaluation metrics.
 * @return CM_Error Error code.
 */
CM_Error evaluate_network(NeuralNetwork *network, float **X_test, float **y_test,
                          int num_samples, int input_size, int output_size,
                          int *metrics, int num_metrics, float *results)
{
    if (network == NULL || X_test == NULL || y_test == NULL || metrics == NULL || results == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (network->head == NULL)
    {
        return CM_LAYER_NOT_INITIALIZED_ERROR;
    }

    /* Calculate the maximum buffer sizes needed for this network */
    int max_input_size, max_output_size;
    CM_Error size_error = calculate_max_buffer_sizes(network, &max_input_size, &max_output_size);
    if (size_error != CM_SUCCESS)
    {
        return size_error;
    }

    /* Ensure max_output_size is at least as large as the provided output_size */
    if (max_output_size < output_size)
    {
        max_output_size = output_size;
    }

    /* Allocate memory for predictions */
    float *predictions = (float *)cm_safe_malloc(max_output_size * sizeof(float), __FILE__, __LINE__);
    if (predictions == NULL)
    {
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < num_metrics; i++)
    {
        results[i] = 0.0f;
    }

    float *y_true = (float *)cm_safe_malloc(num_samples * sizeof(float), __FILE__, __LINE__);
    float *y_pred = (float *)cm_safe_malloc(num_samples * sizeof(float), __FILE__, __LINE__);

    if (y_true == NULL || y_pred == NULL)
    {
        cm_safe_free((void **)&predictions);
        cm_safe_free((void **)&y_true);
        cm_safe_free((void **)&y_pred);
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    for (int i = 0; i < num_samples; i++)
    {
        CM_Error error = forward_pass(network, X_test[i], predictions, input_size, output_size, 0);
        if (error != CM_SUCCESS)
        {
            cm_safe_free((void **)&predictions);
            cm_safe_free((void **)&y_true);
            cm_safe_free((void **)&y_pred);
            return error;
        }

        y_true[i] = y_test[i][0];
        y_pred[i] = predictions[0];
    }

    float threshold = 0.5f;

    for (int i = 0; i < num_metrics; i++)
    {
        switch (metrics[i])
        {
        case METRIC_ACCURACY:
            results[i] = accuracy(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_R2_SCORE:
            results[i] = r2_score(y_true, y_pred, num_samples);
            break;
        case METRIC_PRECISION:
            results[i] = precision(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_RECALL:
            results[i] = recall(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_F1_SCORE:
            results[i] = f1_score(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_BALANCED_ACCURACY:
            results[i] = balanced_accuracy(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_COHENS_KAPPA:
            results[i] = cohens_kappa(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_IOU:
            results[i] = iou(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_MCC:
            results[i] = mcc(y_true, y_pred, num_samples, threshold);
            break;
        case METRIC_MEAN_ABSOLUTE_ERROR:
            results[i] = mean_absolute_error(y_true, y_pred, num_samples);
            break;
        case METRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR:
            results[i] = mean_absolute_percentage_error(y_true, y_pred, num_samples);
            break;
        case METRIC_ROOT_MEAN_SQUARED_ERROR:
            results[i] = root_mean_squared_error(y_true, y_pred, num_samples);
            break;
        case METRIC_SPECIFICITY:
            results[i] = specificity(y_true, y_pred, num_samples, threshold);
            break;
        default:
            LOG_ERROR("Unknown metric type: %d", metrics[i]);
            cm_safe_free((void **)&predictions);
            cm_safe_free((void **)&y_true);
            cm_safe_free((void **)&y_pred);
            return CM_INVALID_PARAMETER_ERROR;
        }
    }

    cm_safe_free((void **)&predictions);
    cm_safe_free((void **)&y_true);
    cm_safe_free((void **)&y_pred);

    return CM_SUCCESS;
}

/**
 * @brief Get the name of a metric type.
 *
 * @param metric Metric type.
 * @return const char* Name of the metric.
 */
const char *get_metric_name(MetricType metric)
{
    switch (metric)
    {
    case METRIC_ACCURACY:
        return "Accuracy";
    case METRIC_BALANCED_ACCURACY:
        return "Balanced Accuracy";
    case METRIC_COHENS_KAPPA:
        return "Cohen's Kappa";
    case METRIC_F1_SCORE:
        return "F1 Score";
    case METRIC_IOU:
        return "Intersection over Union (IoU)";
    case METRIC_MCC:
        return "Matthews Correlation Coefficient (MCC)";
    case METRIC_MEAN_ABSOLUTE_ERROR:
        return "Mean Absolute Error";
    case METRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR:
        return "Mean Absolute Percentage Error";
    case METRIC_PRECISION:
        return "Precision";
    case METRIC_R2_SCORE:
        return "R2 Score";
    case METRIC_RECALL:
        return "Recall";
    case METRIC_ROOT_MEAN_SQUARED_ERROR:
        return "Root Mean Squared Error";
    case METRIC_SPECIFICITY:
        return "Specificity";
    default:
        return "Unknown Metric";
    }
}

/**
 * @brief Get the name of a metric type.
 *
 * @param metric Metric type.
 * @return const char* Name of the metric.
 */
MetricType get_metric_type_from_name(const char *metric_name)
{
    if (strcmp(metric_name, "Accuracy") == 0)
        return METRIC_ACCURACY;
    if (strcmp(metric_name, "R2 Score") == 0)
        return METRIC_R2_SCORE;
    if (strcmp(metric_name, "Precision") == 0)
        return METRIC_PRECISION;
    if (strcmp(metric_name, "Recall") == 0)
        return METRIC_RECALL;
    if (strcmp(metric_name, "F1 Score") == 0)
        return METRIC_F1_SCORE;
    if (strcmp(metric_name, "Balanced Accuracy") == 0)
        return METRIC_BALANCED_ACCURACY;
    if (strcmp(metric_name, "Cohen's Kappa") == 0)
        return METRIC_COHENS_KAPPA;
    if (strcmp(metric_name, "IoU") == 0)
        return METRIC_IOU;
    if (strcmp(metric_name, "MCC") == 0)
        return METRIC_MCC;
    if (strcmp(metric_name, "Mean Absolute Error") == 0)
        return METRIC_MEAN_ABSOLUTE_ERROR;
    if (strcmp(metric_name, "Mean Absolute Percentage Error") == 0)
        return METRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR;
    if (strcmp(metric_name, "Root Mean Squared Error") == 0)
        return METRIC_ROOT_MEAN_SQUARED_ERROR;
    if (strcmp(metric_name, "Specificity") == 0)
        return METRIC_SPECIFICITY;

    return METRIC_NONE; // Unknown metric
}

/**
 * @brief Test the neural network
 *
 * @param network Pointer to the neural network
 * @param X_test Test data
 * @param y_test Test labels
 * @param num_samples Number of test samples
 */
CM_Error test_network(NeuralNetwork *network, float **X_test, float **y_test, int num_samples, ...)
{
    if (network == NULL || X_test == NULL || y_test == NULL)
    {
        LOG_ERROR("Null pointer argument.");
        return CM_NULL_POINTER_ERROR;
    }

    int output_size = 0;
    if (network->tail != NULL)
    {
        switch (network->tail->type)
        {
        case LAYER_DENSE:
            output_size = ((DenseLayer *)network->tail->layer)->output_size;
            break;
        case LAYER_FLATTEN:
            output_size = ((FlattenLayer *)network->tail->layer)->output_size;
            break;
        default:
            LOG_ERROR("Unsupported layer type in the tail.");
            return CM_NOT_IMPLEMENTED_ERROR;
        }
    }

    // Parse optional arguments for metrics
    va_list args;
    va_start(args, num_samples);
    const char **metrics = va_arg(args, const char **);
    va_end(args);

    // Use default metrics if none are provided
    static const char *default_metrics[] = {"R2 Score", NULL};
    if (metrics == NULL)
    {
        metrics = default_metrics;
    }

    // Count the number of metrics
    int num_metrics = 0;
    while (metrics[num_metrics] != NULL)
    {
        num_metrics++;
    }

    int metric_types[num_metrics];
    for (int i = 0; i < num_metrics; i++)
    {
        metric_types[i] = get_metric_type_from_name(metrics[i]);
        if (metric_types[i] == METRIC_NONE)
        {
            LOG_ERROR("Unknown metric '%s'.", metrics[i]);
            return CM_INVALID_PARAMETER_ERROR;
        }
    }

    float results[num_metrics];
    CM_Error error = evaluate_network(network, X_test, y_test, num_samples, network->input_size, output_size, metric_types, num_metrics, results);
    if (error != CM_SUCCESS)
    {
        LOG_ERROR("Failed to evaluate the network. Error code: %d", error);
        return error;
    }

    for (int i = 0; i < num_metrics; i++)
    {
        LOG_INFO("%s: %.4f", metrics[i], results[i]);
    }
    
    return CM_SUCCESS;
}

/**
 * @brief Free memory allocated for the neural network
 *
 * @param network Pointer to the neural network
 * @return CM_Error Error code
 */
CM_Error free_neural_network(NeuralNetwork *network)
{
    if (network == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    NeuralNetworkNode *current = network->head;
    while (current != NULL)
    {
        NeuralNetworkNode *temp = current;
        current = current->next;

        switch (temp->type)
        {
        case LAYER_DENSE:
            free_dense((DenseLayer *)temp->layer);
            break;
        case LAYER_FLATTEN:
            free_flatten((FlattenLayer *)temp->layer);
            break;
        case LAYER_DROPOUT:
            cm_safe_free(&(temp->layer));
            break;
        case LAYER_MAXPOOLING:
            free_maxpooling((MaxPoolingLayer *)temp->layer);
            break;
        case LAYER_POOLING:
            free_polling((PollingLayer *)temp->layer);
            break;
        default:
            cm_safe_free(&(temp->layer));
        }
        
        cm_safe_free(&(temp->layer));
        cm_safe_free((void **)&temp);
    }

    cm_safe_free((void **)&network);

    return CM_SUCCESS;
}

/**
 * @brief Prints a summary of the neural network architecture.
 *
 * @param network Pointer to the neural network.
 */
void summary(NeuralNetwork *network)
{
    if (network == NULL)
    {
        LOG_ERROR("Network is NULL");
        return;
    }

    printf("Neural Network Summary:\n");
    printf("=====================================================================\n");
    printf("%-24s%-14s%-12s%-20s\n", "Layer (type)", "Output Shape", "Param #", "Connected to");
    printf("=====================================================================\n");

    NeuralNetworkNode *current = network->head;
    char prev_name[64] = "input";
    int total_params = 0;
    int layer_index = 1;

    while (current != NULL)
    {
        char layer_name[64] = {0};
        char output_shape[32] = {0};
        int params = 0;

        switch (current->type)
        {
        case LAYER_DENSE:
        {
            DenseLayer *dense = (DenseLayer *)current->layer;
            snprintf(layer_name, sizeof(layer_name), "dense_%d (Dense)", layer_index);
            snprintf(output_shape, sizeof(output_shape), "(None, %d)", dense->output_size);
            params = dense->input_size * dense->output_size + dense->output_size;
            break;
        }
        case LAYER_FLATTEN:
        {
            FlattenLayer *flatten = (FlattenLayer *)current->layer;
            snprintf(layer_name, sizeof(layer_name), "flatten_%d (Flatten)", layer_index);
            snprintf(output_shape, sizeof(output_shape), "(None, %d)", flatten->output_size);
            params = 0;
            break;
        }
        case LAYER_DROPOUT:
        {
            snprintf(layer_name, sizeof(layer_name), "dropout_%d (Dropout)", layer_index);
            snprintf(output_shape, sizeof(output_shape), "(None, ?)");
            params = 0;
            break;
        }
        case LAYER_MAXPOOLING:
        {
            MaxPoolingLayer *mp = (MaxPoolingLayer *)current->layer;
            snprintf(layer_name, sizeof(layer_name), "maxpooling_%d (MaxPooling)", layer_index);
            int out_size = compute_maxpooling_output_size(network->input_size, mp->kernel_size, mp->stride);
            snprintf(output_shape, sizeof(output_shape), "(None, %d)", out_size);
            params = 0;
            break;
        }
        case LAYER_POOLING:
        {
            snprintf(layer_name, sizeof(layer_name), "pooling_%d (Pooling)", layer_index);
            snprintf(output_shape, sizeof(output_shape), "(None, ?)");
            params = 0;
            break;
        }
        default:
        {
            snprintf(layer_name, sizeof(layer_name), "unknown_%d (Unknown)", layer_index);
            snprintf(output_shape, sizeof(output_shape), "(None)");
            params = 0;
            break;
        }
        }

        printf("%-24s%-14s%-12d%-20s\n", layer_name, output_shape, params, prev_name);
        total_params += params;

        if (current->activation != ACTIVATION_NONE)
        {
            char act_name[64] = {0};
            snprintf(act_name, sizeof(act_name), "activation_%d", layer_index);
            printf("%-24s%-14s%-12d%-20s\n", act_name, output_shape, 0, layer_name);
        }

        strncpy(prev_name, layer_name, sizeof(prev_name) - 1);
        prev_name[sizeof(prev_name) - 1] = '\0';

        current = current->next;
        layer_index++;
    }

    printf("=====================================================================\n");
    printf("Total params: %d\n\n", total_params);
}
