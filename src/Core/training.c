#include "../../include/Core/training.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

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
        fprintf(stderr, "[build_network] Error: Null pointer argument.\n");
        return CM_NULL_POINTER_ERROR;
    }

    network->optimizer_type = optimizer_type;
    network->loss_function = loss_function;
    network->learning_rate = learning_rate;
    network->l1_lambda = l1_lambda;
    network->l2_lambda = l2_lambda;

    if (optimizer_type == OPTIMIZER_ADAM)
    {
        network->v_w = (float *)cm_safe_malloc(network->input_size * sizeof(float), __FILE__, __LINE__);
        network->v_b = (float *)cm_safe_malloc(sizeof(float), __FILE__, __LINE__);
        network->s_w = (float *)cm_safe_malloc(network->input_size * sizeof(float), __FILE__, __LINE__);
        network->s_b = (float *)cm_safe_malloc(sizeof(float), __FILE__, __LINE__);

        if (!network->v_w || !network->v_b || !network->s_w || !network->s_b)
        {
            fprintf(stderr, "[build_network] Error: Memory allocation failed for Adam optimizer parameters.\n");
            cm_safe_free((void **)&network->v_w);
            cm_safe_free((void **)&network->v_b);
            cm_safe_free((void **)&network->s_w);
            cm_safe_free((void **)&network->s_b);
            return CM_MEMORY_ALLOCATION_ERROR;
        }

        memset(network->v_w, 0, network->input_size * sizeof(float));
        memset(network->v_b, 0, sizeof(float));
        memset(network->s_w, 0, network->input_size * sizeof(float));
        memset(network->s_b, 0, sizeof(float));
    }
    else if (optimizer_type == OPTIMIZER_RMSPROP)
    {
        network->cache_w = (float *)cm_safe_malloc(network->input_size * sizeof(float), __FILE__, __LINE__);
        network->cache_b = (float *)cm_safe_malloc(sizeof(float), __FILE__, __LINE__);

        if (!network->cache_w || !network->cache_b)
        {
            fprintf(stderr, "[build_network] Error: Memory allocation failed for RMSProp optimizer parameters.\n");
            cm_safe_free((void **)&network->cache_w);
            cm_safe_free((void **)&network->cache_b);
            return CM_MEMORY_ALLOCATION_ERROR;
        }

        memset(network->cache_w, 0, network->input_size * sizeof(float));
        memset(network->cache_b, 0, sizeof(float));
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
            fprintf(stderr, "[apply_activation_array] Error: Softmax activation failed.\n");
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
    LayerConfig config;
    config.type = type;
    config.activation = activation;

    switch (type)
    {
    case LAYER_DENSE:
        config.params.dense.input_size = input_size;
        config.params.dense.output_size = output_size;
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
        fprintf(stderr, "[model_add] Error: Unsupported layer type.\n");
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

    float *layer_input = (float *)cm_safe_malloc(input_size * sizeof(float), __FILE__, __LINE__);
    if (layer_input == NULL)
    {
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    int max_size = (input_size > output_size) ? input_size : output_size;
    float *layer_output = (float *)cm_safe_malloc(max_size * sizeof(float), __FILE__, __LINE__);
    if (layer_output == NULL)
    {
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
                fprintf(stderr, "[forward_pass] Error: Input Size mismatch - current: %d layer: %d\n", current_size, dense->input_size);
                dense->input_size = current_size;
            }
            if (dense->output_size != output_size)
            {
                fprintf(stderr, "[forward_pass] Error: Output Size mismatch - current: %d layer: %d\n", output_size, dense->output_size);
                dense->output_size = output_size;
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
        fprintf(stderr, "[calculate_loss] Error: Unknown loss type.\n");
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
        fprintf(stderr, "[calculate_loss_gradient] Error: Unknown loss type.\n");
        break;
    }
}

/**
 * @brief Train the neural network
 *
 * @param network Pointer to the neural network
 * @param X_train Training data
 * @param y_train Training labels
 * @param num_samples Number of training samples
 * @param input_size Size of each input sample
 * @param output_size Size of each output sample
 * @param batch_size Batch size for training
 * @param epochs Number of training epochs
 * @return CM_Error Error code
 */
CM_Error train_network(NeuralNetwork *network, float **X_train, float **y_train,
                       int num_samples, int input_size, int output_size,
                       int batch_size, int epochs)
{
    if (network == NULL || X_train == NULL || y_train == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (network->head == NULL)
    {
        return CM_LAYER_NOT_INITIALIZED_ERROR;
    }

    if (network->optimizer_type == OPTIMIZER_NONE)
    {
        fprintf(stderr, "[train_network] Error: Optimizer is not set. Call build_network first.\n");
        return CM_OPTIMIZER_NOT_INITIALIZED_ERROR;
    }

    float *predictions = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);
    if (predictions == NULL)
    {
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    float *loss_gradient = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);
    if (loss_gradient == NULL)
    {
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

                CM_Error error = forward_pass(network, X_train[idx], predictions, input_size, output_size, 1);
                if (error != CM_SUCCESS)
                {
                    cm_safe_free((void **)&predictions);
                    cm_safe_free((void **)&loss_gradient);
                    return error;
                }

                float loss = calculate_loss(predictions, y_train[idx], output_size, network->loss_function);
                total_loss += loss;

                calculate_loss_gradient(predictions, y_train[idx], loss_gradient, output_size, network->loss_function);

                NeuralNetworkNode *current = network->head;
                int layer_index = 0; // compiler warning:  warning: variable 'layer_index' set but not used

                while (current != NULL)
                {
                    if (current->type == LAYER_DENSE)
                    {
                        DenseLayer *dense = (DenseLayer *)current->layer;

                        if (network->l1_lambda > 0.0f || network->l2_lambda > 0.0f)
                        {
                            l1_l2(dense->weights, loss_gradient, network->l1_lambda, network->l2_lambda, dense->input_size * dense->output_size);
                        }

                        for (int i = 0; i < dense->output_size; i++)
                        {
                            for (int j = 0; j < dense->input_size; j++)
                            {
                                float x = X_train[idx][j]; 
                                float *w = &dense->weights[i * dense->input_size + j];
                                float *b = &dense->biases[i];
                                float lr = network->learning_rate;

                                switch (network->optimizer_type)
                                {
                                case OPTIMIZER_SGD:
                                    *w -= lr * loss_gradient[i] * x;
                                    *b -= lr * loss_gradient[i];
                                    break;
                                case OPTIMIZER_RMSPROP:
                                {
                                    float *cache_w = &network->cache_w[i * dense->input_size + j];
                                    float *cache_b = &network->cache_b[i];

                                    *cache_w = network->beta1 * (*cache_w) +
                                               (1 - network->beta1) * pow(loss_gradient[i] * x, 2);
                                    *cache_b = network->beta1 * (*cache_b) +
                                               (1 - network->beta1) * pow(loss_gradient[i], 2);

                                    *w -= lr * (loss_gradient[i] * x) / (sqrt(*cache_w) + network->epsilon);
                                    *b -= lr * loss_gradient[i] / (sqrt(*cache_b) + network->epsilon);
                                    break;
                                }
                                case OPTIMIZER_ADAM:
                                {
                                    float *v_w = &network->v_w[i * dense->input_size + j];
                                    float *v_b = &network->v_b[i];
                                    float *s_w = &network->s_w[i * dense->input_size + j];
                                    float *s_b = &network->s_b[i];

                                    *v_w = network->beta1 * (*v_w) + (1 - network->beta1) * (loss_gradient[i] * x);
                                    *v_b = network->beta1 * (*v_b) + (1 - network->beta1) * loss_gradient[i];

                                    *s_w = network->beta2 * (*s_w) + (1 - network->beta2) * pow(loss_gradient[i] * x, 2);
                                    *s_b = network->beta2 * (*s_b) + (1 - network->beta2) * pow(loss_gradient[i], 2);

                                    float v_w_corrected = *v_w / (1 - pow(network->beta1, epoch + 1));
                                    float v_b_corrected = *v_b / (1 - pow(network->beta1, epoch + 1));

                                    float s_w_corrected = *s_w / (1 - pow(network->beta2, epoch + 1));
                                    float s_b_corrected = *s_b / (1 - pow(network->beta2, epoch + 1));

                                    *w -= lr * v_w_corrected / (sqrt(s_w_corrected) + network->epsilon);
                                    *b -= lr * v_b_corrected / (sqrt(s_b_corrected) + network->epsilon);
                                    break;
                                }
                                default:
                                    fprintf(stderr, "[train_network] Error: Unknown optimizer type.\n");
                                    cm_safe_free((void **)&predictions);
                                    cm_safe_free((void **)&loss_gradient);
                                    return CM_INVALID_PARAMETER_ERROR;
                                }
                            }
                        }
                    }
                    current = current->next;
                    layer_index++;
                }
            }
        }

        printf("[train_network] Info: Epoch %d/%d - Loss: %.4f\n", epoch + 1, epochs, total_loss / num_samples);
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

    float *predictions = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);
    if (predictions == NULL)
    {
        return CM_MEMORY_ALLOCATION_ERROR;
    }

    float *total_loss = (float *)cm_safe_malloc(sizeof(float), __FILE__, __LINE__);
    if (total_loss == NULL)
    {
        cm_safe_free((void **)&predictions);
        return CM_MEMORY_ALLOCATION_ERROR;
    }
    *total_loss = 0.0f;

    for (int i = 0; i < num_metrics; i++)
    {
        results[i] = 0.0f;
    }

    float *y_true = (float *)cm_safe_malloc(num_samples * sizeof(float), __FILE__, __LINE__);
    float *y_pred = (float *)cm_safe_malloc(num_samples * sizeof(float), __FILE__, __LINE__);

    if (y_true == NULL || y_pred == NULL)
    {
        cm_safe_free((void **)&predictions);
        cm_safe_free((void **)&total_loss);
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
            cm_safe_free((void **)&total_loss);
            cm_safe_free((void **)&y_true);
            cm_safe_free((void **)&y_pred);
            return error;
        }

        float loss = calculate_loss(predictions, y_test[i], output_size, network->loss_function);
        *total_loss += loss;

        y_true[i] = y_test[i][0];
        y_pred[i] = predictions[0];
    }

    for (int i = 0; i < num_metrics; i++)
    {
        switch (metrics[i])
        {
        case METRIC_ACCURACY:
        {
            float threshold = 0.1f;
            int correct_predictions = 0;
            for (int j = 0; j < num_samples; j++)
            {
                if (fabs(y_pred[j] - y_true[j]) < threshold)
                {
                    correct_predictions++;
                }
            }
            results[i] = (float)correct_predictions / num_samples;
            break;
        }
        case METRIC_R2_SCORE:
            results[i] = r2_score(y_true, y_pred, num_samples);
            break;
        default:
            fprintf(stderr, "[evaluate_network] Error: Unknown metric type.\n");
            cm_safe_free((void **)&predictions);
            cm_safe_free((void **)&total_loss);
            cm_safe_free((void **)&y_true);
            cm_safe_free((void **)&y_pred);
            return CM_INVALID_PARAMETER_ERROR;
        }
    }

    *total_loss /= num_samples;

    cm_safe_free((void **)&predictions);
    cm_safe_free((void **)&total_loss);
    cm_safe_free((void **)&y_true);
    cm_safe_free((void **)&y_pred);

    return CM_SUCCESS;
}

/**
 * @brief Test the neural network
 *
 * @param network Pointer to the neural network
 * @param X_test Test data
 * @param y_test Test labels
 * @param num_samples Number of test samples
 * @param input_size Size of each input sample
 * @param output_size Size of each output sample
 * @param metrics Array of evaluation metrics to calculate.
 * @param num_metrics Number of evaluation metrics to calculate.
 * @param results Array to store the results of the evaluation metrics.
 * @return CM_Error Error code
 */
CM_Error test_network(NeuralNetwork *network, float **X_test, float **y_test,
                      int num_samples, int input_size, int output_size,
                      int *metrics, int num_metrics, float *results)
{
    return evaluate_network(network, X_test, y_test, num_samples, input_size, output_size, metrics, num_metrics, results);
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
        printf("[summary] Error: Network is NULL.\n");
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