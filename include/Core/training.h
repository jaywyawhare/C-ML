#ifndef C_ML_TRAINING_H
#define C_ML_TRAINING_H

#include <stddef.h>
#include "../Core/error_codes.h"
#include "../Layers/dense.h"
#include "../Layers/flatten.h"
#include "../Layers/dropout.h"

#include "../Core/memory_management.h"

#include "../Optimizers/adam.h"
#include "../Optimizers/rmsprop.h"
#include "../Optimizers/sgd.h"

#include "../Metrics/accuracy.h"
#include "../Metrics/balanced_accuracy.h"
#include "../Metrics/cohens_kappa.h"
#include "../Metrics/f1_score.h"
#include "../Metrics/iou.h"
#include "../Metrics/mcc.h"
#include "../Metrics/mean_absolute_error.h"
#include "../Metrics/mean_absolute_percentage_error.h"
#include "../Metrics/precision.h"
#include "../Metrics/r2_score.h"
#include "../Metrics/recall.h"
#include "../Metrics/reduce_mean.h"
#include "../Metrics/root_mean_squared_error.h"
#include "../Metrics/specificity.h"

#include "../Loss_Functions/binary_cross_entropy_loss.h"
#include "../Loss_Functions/cosine_similarity_loss.h"
#include "../Loss_Functions/focal_loss.h"
#include "../Loss_Functions/huber_loss.h"
#include "../Loss_Functions/kld_loss.h"
#include "../Loss_Functions/log_cosh_loss.h"
#include "../Loss_Functions/mean_squared_error.h"
#include "../Loss_Functions/poisson_loss.h"
#include "../Loss_Functions/smooth_l1_loss.h"
#include "../Loss_Functions/tversky_loss.h"

#include "../Core/dataset.h" 

/**
 * @brief Enumeration for evaluation metrics
 */
typedef enum
{
    METRIC_NONE,
    METRIC_ACCURACY,
    METRIC_BALANCED_ACCURACY,
    METRIC_COHENS_KAPPA,
    METRIC_F1_SCORE,
    METRIC_IOU,
    METRIC_MCC,
    METRIC_MEAN_ABSOLUTE_ERROR,
    METRIC_MEAN_ABSOLUTE_PERCENTAGE_ERROR,
    METRIC_PRECISION,
    METRIC_R2_SCORE,
    METRIC_RECALL,
    METRIC_ROOT_MEAN_SQUARED_ERROR,
    METRIC_SPECIFICITY
} MetricType;

/**
 * @brief Enumeration for layer types
 */
typedef enum
{
    LAYER_NONE,
    LAYER_DENSE,
    LAYER_FLATTEN,
    LAYER_DROPOUT,
    LAYER_MAXPOOLING,
    LAYER_POOLING
} LayerType;

/**
 * @brief Enumeration for activation functions
 */
typedef enum
{
    ACTIVATION_NONE,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX,
    ACTIVATION_LEAKY_RELU,
    ACTIVATION_ELU,
    ACTIVATION_GELU,
    ACTIVATION_LINEAR
} ActivationType;

/**
 * @brief Enumeration for loss function types
 */
typedef enum
{
    LOSS_NONE,
    LOSS_MSE,
    LOSS_BINARY_CROSS_ENTROPY,
    LOSS_FOCAL,
    LOSS_HUBER,
    LOSS_KLD,
    LOSS_LOG_COSH,
    LOSS_POISSON,
    LOSS_SMOOTH_L1,
    LOSS_TVERSKY,
    LOSS_COSINE_SIMILARITY
} LossType;

/**
 * @brief Enumeration for optimizer types
 */
typedef enum
{
    OPTIMIZER_NONE,
    OPTIMIZER_SGD,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM
} OptimizerType;

/**
 * @brief Structure for layer configuration
 */
typedef struct
{
    LayerType type;
    ActivationType activation;
    union
    {
        struct
        {
            int input_size;
            int output_size;
        } dense;
        struct
        {
            float rate;
        } dropout;
        struct
        {
            int kernel_size;
            int stride;
        } pooling;
    } params;
} LayerConfig;

/**
 * @brief Structure representing a node in a neural network
 */
typedef struct NeuralNetworkNode
{
    LayerType type;
    ActivationType activation;
    void *layer;
    struct NeuralNetworkNode *next;
} NeuralNetworkNode;

/**
 * @brief Structure representing a neural network
 */
typedef struct
{
    NeuralNetworkNode *head;
    NeuralNetworkNode *tail;
    int num_layers;
    int input_size;
    int max_layer_output_size;  /* Maximum output size of any layer in the network */
    OptimizerType optimizer_type;
    int loss_function;
    float learning_rate;
    float l1_lambda;
    float l2_lambda;
    float beta1;
    float beta2;
    float epsilon;
    float *cache_w;
    float *cache_b;
    float *v_w;
    float *v_b;
    float *s_w;
    float *s_b;
} NeuralNetwork;

/**
 * @brief Initialize a neural network
 *
 * @param input_size The size of the input layer
 * @return NeuralNetwork* Pointer to the initialized neural network
 */
NeuralNetwork *create_neural_network(int input_size);

/**
 * @brief Build the neural network by setting the optimizer and loss function.
 *
 * @param network Pointer to the neural network.
 * @param optimizer_type The type of optimization algorithm to use.
 * @param learning_rate The learning rate for the optimizer.
 * @param loss_function The loss function to use.
 * @param l1_lambda L1 regularization parameter
 * @param l2_lambda L2 regularization parameter
 * @return CM_Error Error code.
 */
CM_Error build_network(NeuralNetwork *network, OptimizerType optimizer_type, float learning_rate, int loss_function, float l1_lambda, float l2_lambda);

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
CM_Error calculate_max_buffer_sizes(NeuralNetwork *network, int *max_input_size, int *max_output_size);

/**
 * @brief Initialize optimizer parameters based on the current network structure.
 * 
 * This function allocates memory for optimizer parameters based on the maximum layer size.
 * It should be called before training starts, after all layers have been added.
 *
 * @param network Pointer to the neural network.
 * @return CM_Error Error code.
 */
CM_Error initialize_optimizer_params(NeuralNetwork *network);

/**
 * @brief Add a layer to the neural network based on configuration
 *
 * @param network Pointer to the neural network
 * @param config Layer configuration
 * @return CM_Error Error code
 */
CM_Error add_layer(NeuralNetwork *network, LayerConfig config);

/**
 * @brief Adds a layer to the neural network using a simplified interface.
 *
 * @param network Pointer to the neural network.
 * @param type The type of layer to add.
 * @param activation The activation function to use.
 * @param input_size The input size for dense layers.
 * @param output_size The output size for dense layers.
 * @param rate The dropout rate for dropout layers.
 * @param kernel_size The kernel size for pooling layers.
 * @param stride The stride for pooling layers.
 * @return CM_Error Error code.
 */
CM_Error model_add(NeuralNetwork *network, LayerType type, ActivationType activation,
                   int input_size, int output_size, float rate, int kernel_size, int stride);

/**
 * @brief Forward pass through the neural network
 *
 * @param network Pointer to the neural network
 * @param input Input data
 * @param output Output data
 * @param input_size Size of the input
 * @param output_size Size of the output
 * @param is_training Whether in training mode (affects dropout)
 * @return CM_Error Error code
 */
CM_Error forward_pass(NeuralNetwork *network, float *input, float *output, int input_size, int output_size, int is_training);

/**
 * @brief Train the neural network.
 *
 * @param network Pointer to the neural network.
 * @param dataset Pointer to the dataset structure.
 * @param epochs Number of training epochs.
 * @param ... Optional argument for batch size (default is 1).
 * @return CM_Error Error code.
 */
CM_Error train_network(NeuralNetwork *network, Dataset *dataset, int epochs, ...);

/**
 * @brief Test the neural network
 *
 * @param network Pointer to the neural network
 * @param X_test Test data
 * @param y_test Test labels
 * @param num_samples Number of test samples
 * @param ... Optional arguments: metrics (const char**, terminated by NULL)
 */
void test_network(NeuralNetwork *network, float **X_test, float **y_test, int num_samples, ...);

/**
 * @brief Free memory allocated for the neural network
 *
 * @param network Pointer to the neural network
 * @return CM_Error Error code
 */
CM_Error free_neural_network(NeuralNetwork *network);

/**
 * @brief Calculate the loss between predicted and actual values
 *
 * @param predicted Pointer to predicted values
 * @param actual Pointer to actual values
 * @param size Number of values
 * @param loss_type Type of loss function to use
 * @return float Calculated loss
 */
float calculate_loss(float *predicted, float *actual, int size, LossType loss_type);

/**
 * @brief Calculate the gradient of the loss function
 *
 * @param predicted Pointer to predicted values
 * @param actual Pointer to actual values
 * @param gradient Pointer to store the calculated gradient
 * @param size Number of values
 * @param loss_type Type of loss function
 */
void calculate_loss_gradient(float *predicted, float *actual, float *gradient, int size, LossType loss_type);

/**
 * @brief Prints a summary of the neural network
 *
 * @param network Pointer to the neural network
 */
void summary(NeuralNetwork *network);

/**
 * @brief Get the name of a metric type
 *
 * @param metric The metric type
 * @return const char* The name of the metric
 */
const char *get_metric_name(MetricType metric);

#endif
