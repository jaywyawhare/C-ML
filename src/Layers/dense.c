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

    for (int i = 0; i < layer->output_size; i++)
    {
        Node *sum = tensor(0.0f, 1);
        for (int j = 0; j < layer->input_size; j++)
        {
            Node *x = tensor(input[j], 1);
            Node *w = tensor(layer->weights[j + i * layer->input_size], 1);
            sum = add(sum, mul(x, w));
            cm_safe_free((void **)&x);
            cm_safe_free((void **)&w);
        }
        Node *b = tensor(layer->biases[i], 1);
        output[i] = add(sum, b)->value;
        cm_safe_free((void **)&sum);
        cm_safe_free((void **)&b);
    }
    return CM_SUCCESS;
}

int backward_dense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input)
{
    if (!layer || !input || !output || !d_output || !d_input)
        return CM_NULL_POINTER_ERROR;

    for (int i = 0; i < layer->input_size; i++)
    {
        Node *x = tensor(input[i], 1);
        d_input[i] = 0.0f;

        for (int j = 0; j < layer->output_size; j++)
        {
            Node *dy = tensor(d_output[j], 1);
            Node *w = tensor(layer->weights[i * layer->output_size + j], 1);
            int idx = i * layer->output_size + j;

            // Compute gradients and apply optimizer updates
            Node *dw = mul(x, dy);
            float grad = dw->value;

            switch (layer->optimizer_type)
            {
            case OPTIMIZER_ADAM:
                adam(x->value, output[j], layer->learning_rate,
                     &layer->weights[idx], &layer->biases[j],
                     layer->adam_m_w + idx, layer->adam_m_b + j,
                     layer->adam_v_w + idx, layer->adam_v_b + j,
                     NULL, NULL, layer->adam_config, layer->step);
                break;

            case OPTIMIZER_RMSPROP:
                rmsprop(x->value, output[j], layer->learning_rate,
                        &layer->weights[idx], &layer->biases[j],
                        layer->rms_cache_w + idx, layer->rms_cache_b + j,
                        NULL, NULL, NULL, NULL,
                        layer->rmsprop_config, layer->step);
                break;

            case OPTIMIZER_SGD:
                sgd(x->value, output[j], layer->learning_rate,
                    &layer->weights[idx], &layer->biases[j],
                    NULL, NULL, layer->sgd_config, layer->step);
                break;

            default:
                LOG_ERROR("Invalid optimizer type: %d", layer->optimizer_type);
                return CM_INVALID_OPTIMIZER_ERROR;
            }

            d_input[i] += mul(w, dy)->value;
            cm_safe_free((void **)&dy);
            cm_safe_free((void **)&w);
            cm_safe_free((void **)&dw);
        }
        cm_safe_free((void **)&x);
    }

    layer->step++;
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
    {
        cm_safe_free((void **)&layer->weights);
    }
    if (layer->biases != NULL)
    {
        cm_safe_free((void **)&layer->biases);
    }
    if (layer->adam_m_w != NULL)
    {
        cm_safe_free((void **)&layer->adam_m_w);
    }
    if (layer->adam_m_b != NULL)
    {
        cm_safe_free((void **)&layer->adam_m_b);
    }
    if (layer->adam_v_w != NULL)
    {
        cm_safe_free((void **)&layer->adam_v_w);
    }
    if (layer->adam_v_b != NULL)
    {
        cm_safe_free((void **)&layer->adam_v_b);
    }
    if (layer->rms_cache_w != NULL)
    {
        cm_safe_free((void **)&layer->rms_cache_w);
    }
    if (layer->rms_cache_b != NULL)
    {
        cm_safe_free((void **)&layer->rms_cache_b);
    }
    return CM_SUCCESS;
}
