#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "../../include/Core/autograd.h"
#include "../../include/Optimizers/optimizer_types.h"
#include "../../include/Optimizers/adam.h"
#include "../../include/Optimizers/rmsprop.h"
#include "../../include/Optimizers/sgd.h"
#include "../../include/Layers/dense.h"
#include "../../include/Core/error_codes.h"
#include "../../include/Core/memory_management.h"
#include "../../include/Core/logging.h"

/**
 * @brief Initializes a Dense Layer with random weights and biases.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input_size Number of input neurons.
 * @param output_size Number of output neurons.
 * @return int Error code.
 */
int initialize_dense(DenseLayer *layer, int input_size, int output_size)
{
    if (layer == NULL)
    {
        LOG_ERROR("Layer is NULL");
        return CM_NULL_POINTER_ERROR;
    }

    if (input_size <= 0 || output_size <= 0)
    {
        LOG_ERROR("Invalid input size (%d) or output size (%d)", input_size, output_size);
        return CM_INVALID_PARAMETER_ERROR;
    }

    // Initialize weights and biases
    layer->weights = (float *)cm_safe_malloc(input_size * output_size * sizeof(float), __FILE__, __LINE__);
    layer->biases = (float *)cm_safe_malloc(output_size * sizeof(float), __FILE__, __LINE__);

    // Initialize optimizer states
    layer->adam_m_w = (float *)cm_safe_calloc(input_size * output_size, sizeof(float), __FILE__, __LINE__);
    layer->adam_m_b = (float *)cm_safe_calloc(output_size, sizeof(float), __FILE__, __LINE__);
    layer->adam_v_w = (float *)cm_safe_calloc(input_size * output_size, sizeof(float), __FILE__, __LINE__);
    layer->adam_v_b = (float *)cm_safe_calloc(output_size, sizeof(float), __FILE__, __LINE__);

    layer->rms_cache_w = (float *)cm_safe_calloc(input_size * output_size, sizeof(float), __FILE__, __LINE__);
    layer->rms_cache_b = (float *)cm_safe_calloc(output_size, sizeof(float), __FILE__, __LINE__);

    // Initialize default optimizer settings
    layer->optimizer_type = OPTIMIZER_SGD;
    layer->learning_rate = 0.01f;
    layer->step = 0;

    LOG_DEBUG("Initialized DenseLayer with input size (%d) and output size (%d)", input_size, output_size);

    for (int i = 0; i < input_size * output_size; i++)
    {
        layer->weights[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    for (int i = 0; i < output_size; i++)
    {
        layer->biases[i] = ((float)rand() / RAND_MAX) - 0.5;
    }

    return CM_SUCCESS;
}

/**
 * @brief Performs the forward pass for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @param input Input data array.
 * @param output Output data array.
 * @return int Error code.
 */
int forward_dense(DenseLayer *layer, float *input, float *output)
{
    if (!layer || !input || !output)
    {
        LOG_ERROR("Layer, input, or output is NULL");
        return CM_NULL_POINTER_ERROR;
    }

    float *output_temp = (float *)cm_safe_malloc(layer->output_size * sizeof(float), __FILE__, __LINE__);
    if (!output_temp)
        return CM_MEMORY_ALLOCATION_ERROR;

    for (int i = 0; i < layer->output_size; i++)
    {
        Node *sum = tensor(0.0f, 1);
        for (int j = 0; j < layer->input_size; j++)
        {
            Node *x = tensor(input[j], 0);
            Node *w = tensor(layer->weights[i * layer->input_size + j], 0);
            Node *prod = tensor_mul(x, w);
            Node *new_sum = tensor_add(sum, prod);
            cm_safe_free((void **)&x);
            cm_safe_free((void **)&w);
            cm_safe_free((void **)&prod);
            cm_safe_free((void **)&sum);
            sum = new_sum;
        }
        Node *b = tensor(layer->biases[i], 0);
        Node *sum_b = tensor_add(sum, b);
        output_temp[i] = sum_b->tensor->storage->data[0];
        cm_safe_free((void **)&sum);
        cm_safe_free((void **)&b);
        cm_safe_free((void **)&sum_b);
    }

    memcpy(output, output_temp, layer->output_size * sizeof(float));
    cm_safe_free((void **)&output_temp);

    return CM_SUCCESS;
}

int backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (!layer || !input || !output || !d_output || !d_input)
        return CM_NULL_POINTER_ERROR;

    memset(d_input, 0, layer->input_size * sizeof(float));

    for (int i = 0; i < layer->input_size; i++)
    {
        for (int j = 0; j < layer->output_size; j++)
        {
            float x_val = input[i];
            float dy_val = d_output[j];
            int weight_idx = j * layer->input_size + i;

            switch (layer->optimizer_type)
            {
            case OPTIMIZER_ADAM:
                adam(x_val, output[j], layer->learning_rate,
                     &layer->weights[weight_idx], &layer->biases[j],
                     &layer->adam_m_w[weight_idx], &layer->adam_m_b[j],
                     &layer->adam_v_w[weight_idx], &layer->adam_v_b[j],
                     NULL, NULL,
                     layer->adam_config, layer->step);
                break;

            case OPTIMIZER_RMSPROP:
                rmsprop(x_val, output[j], layer->learning_rate,
                        &layer->weights[weight_idx], &layer->biases[j],
                        &layer->rms_cache_w[weight_idx], &layer->rms_cache_b[j],
                        NULL, NULL, // v_w, v_b not used
                        NULL, NULL, // avg_grad_w, avg_grad_b not used
                        layer->rmsprop_config,
                        layer->step);
                break;

            case OPTIMIZER_SGD:
                sgd(x_val, output[j], layer->learning_rate,
                    &layer->weights[weight_idx], &layer->biases[j],
                    NULL, NULL,
                    layer->sgd_config, layer->step);
                break;

            default:
                return CM_INVALID_OPTIMIZER_ERROR;
            }

            Node *w_node = tensor(layer->weights[weight_idx], 1);
            Node *dy_node = tensor(dy_val, 1);
            Node *grad = tensor_mul(w_node, dy_node);
            d_input[i] += grad->tensor->storage->data[0];

            cm_safe_free((void **)&w_node);
            cm_safe_free((void **)&dy_node);
            cm_safe_free((void **)&grad);
        }
    }

    return CM_SUCCESS;
}

/**
 * @brief Frees the memory allocated for the Dense Layer.
 *
 * @param layer Pointer to the DenseLayer structure.
 * @return int Error code.
 */
int free_dense(DenseLayer *layer)
{
    if (layer == NULL)
    {
        return CM_NULL_POINTER_ERROR;
    }

    if (layer->weights != NULL)
        cm_safe_free((void **)&layer->weights);
    if (layer->biases != NULL)
        cm_safe_free((void **)&layer->biases);
    if (layer->adam_m_w != NULL)
        cm_safe_free((void **)&layer->adam_m_w);
    if (layer->adam_m_b != NULL)
        cm_safe_free((void **)&layer->adam_m_b);
    if (layer->adam_v_w != NULL)
        cm_safe_free((void **)&layer->adam_v_w);
    if (layer->adam_v_b != NULL)
        cm_safe_free((void **)&layer->adam_v_b);
    if (layer->rms_cache_w != NULL)
        cm_safe_free((void **)&layer->rms_cache_w);
    if (layer->rms_cache_b != NULL)
        cm_safe_free((void **)&layer->rms_cache_b);

    return CM_SUCCESS;
}
