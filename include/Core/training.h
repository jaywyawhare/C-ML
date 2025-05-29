#ifndef C_ML_TRAINING_H
#define C_ML_TRAINING_H

#include "autograd.h"
#include "dataset.h"
#include "error_codes.h"

// Forward declarations
typedef struct NeuralNetwork NeuralNetwork;
typedef struct Layer Layer;
typedef struct OptimizerState OptimizerState;

// Enums for configuration
typedef enum {
    LAYER_DENSE,
    LAYER_CONV2D,
    LAYER_MAXPOOL,
    LAYER_DROPOUT,
    LAYER_FLATTEN,
    LAYER_BATCHNORM
} LayerType;

typedef enum {
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

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_ADAM,
    OPTIMIZER_RMSPROP
} OptimizerType;

typedef enum {
    LOSS_MSE,
    LOSS_BINARY_CROSS_ENTROPY,
    LOSS_CATEGORICAL_CROSS_ENTROPY,
    LOSS_HUBER,
    LOSS_FOCAL
} LossType;

// Layer configuration
typedef struct {
    LayerType type;
    ActivationType activation;
    int input_size;
    int output_size;
    float dropout_rate;
    int kernel_size;
    int stride;
    int padding;
} LayerConfig;

// Layer structure with PyTorch-like parameters
typedef struct Layer {
    LayerType type;
    ActivationType activation;
    
    // Parameters as autograd nodes (PyTorch-like)
    Node *weights;
    Node *bias;
    
    // Layer dimensions
    int input_size;
    int output_size;
    
    // Activation cache for backward pass
    Node *last_input;
    Node *last_output;
    
    // Dropout specific
    float dropout_rate;
    Node *dropout_mask;
    
    // Conv2D specific
    int kernel_size;
    int stride;
    int padding;
    
    struct Layer *next;
} Layer;

// Optimizer state
typedef struct OptimizerState {
    OptimizerType type;
    float learning_rate;
    
    // Adam specific
    Node **m_weights;  // First moment estimates for weights
    Node **v_weights;  // Second moment estimates for weights
    Node **m_bias;     // First moment estimates for bias
    Node **v_bias;     // Second moment estimates for bias
    float beta1;
    float beta2;
    float epsilon;
    int t;  // Time step
    
    // SGD with momentum
    Node **momentum_weights;
    Node **momentum_bias;
    float momentum;
    
    // RMSprop specific
    Node **sq_avg_weights;
    Node **sq_avg_bias;
    float alpha;
    
    // Regularization
    float l1_lambda;
    float l2_lambda;
} OptimizerState;

// Neural Network structure
typedef struct NeuralNetwork {
    Layer *layers;
    int num_layers;
    int input_size;
    
    OptimizerState *optimizer;
    LossType loss_function;
    
    // Training state
    int is_training;
    float last_loss;
    
    // PyTorch-like module registration
    Node **parameters;
    int num_parameters;
} NeuralNetwork;

// Core training functions
NeuralNetwork *create_neural_network(int input_size);
CM_Error build_network(NeuralNetwork *network, OptimizerType optimizer_type, 
                      float learning_rate, LossType loss_function, 
                      float l1_lambda, float l2_lambda);
CM_Error model_add(NeuralNetwork *network, LayerType type, ActivationType activation,
                  int input_size, int output_size, float rate, int kernel_size, int stride);
CM_Error add_layer(NeuralNetwork *network, LayerConfig config);

// Forward and backward pass
CM_Error forward_pass(NeuralNetwork *network, float *input, float *output, 
                     int input_size, int output_size, int is_training);
Node *forward_network(NeuralNetwork *network, Node *input);
CM_Error backward_pass(NeuralNetwork *network, Node *loss);

// Training and evaluation
CM_Error train_network(NeuralNetwork *network, Dataset *dataset, int epochs);
CM_Error train_step(NeuralNetwork *network, Node *input, Node *target);
CM_Error test_network(NeuralNetwork *network, float **X_test, float **y_test, 
                     int num_samples, float *accuracy);
CM_Error evaluate_network(NeuralNetwork *network, Dataset *dataset, float *metrics);

// Loss functions (PyTorch-like) - use different names to avoid autograd conflicts
Node *mse_loss_training(Node *predicted, Node *target);
Node *binary_cross_entropy_loss_training(Node *predicted, Node *target);
Node *categorical_cross_entropy_loss_training(Node *predicted, Node *target);

// Optimizer functions
CM_Error initialize_optimizer_params(NeuralNetwork *network);
CM_Error optimizer_step(NeuralNetwork *network);
CM_Error zero_grad_network(NeuralNetwork *network);

// Utility functions
void summary(NeuralNetwork *network);
void free_neural_network(NeuralNetwork *network);

// Parameter management (PyTorch-like)
CM_Error register_parameter(NeuralNetwork *network, Node *param);
void apply_regularization(NeuralNetwork *network);

#endif // C_ML_TRAINING_H
