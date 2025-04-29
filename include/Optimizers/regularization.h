#ifndef REGULARIZATION_H
#define REGULARIZATION_H

#include "../Core/autograd.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

typedef enum
{
    NO_REGULARIZATION,
    L1_REGULARIZATION,
    L2_REGULARIZATION,
    ELASTIC_NET,
    SPARSE_REGULARIZATION
} RegularizationType;

typedef enum
{
    STEP_LR,
    EXPONENTIAL_LR,
    COSINE_LR
} LRSchedulerType;

typedef enum
{
    WEIGHT_DECAY_NONE,
    WEIGHT_DECAY_L1,
    WEIGHT_DECAY_L2,
    WEIGHT_DECAY_DECOUPLED
} WeightDecayMode;

typedef struct
{
    RegularizationType type;
    float lambda;
    float l1_ratio;
    bool decay_only_weights;
    WeightDecayMode decay_mode;
    float accumulator;
    float scale_factor;
} RegularizerConfig;

Node *compute_regularization_node(Node *param, RegularizerConfig config);
Node *apply_regularization_node(Node *param, RegularizerConfig config);
Node *apply_decoupled_weight_decay_node(Node *param, float weight_decay, float lr);
Node *adjust_learning_rate_node(Node *lr, int epoch, LRSchedulerType scheduler_type,
                                float gamma, int step_size);

#endif
