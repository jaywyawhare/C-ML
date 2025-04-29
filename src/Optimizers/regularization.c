#include "../../include/Optimizers/regularization.h"
#include "../../include/Core/autograd.h"

Node *compute_regularization_node(Node *param, RegularizerConfig config)
{
    Node *result = NULL;
    switch (config.type)
    {
    case L1_REGULARIZATION:
    {
        Node *abs_param = pow(pow(param, tensor(2.0f, 0)), tensor(0.5f, 0));
        result = mul(tensor(config.lambda, 0), abs_param);
        break;
    }
    case L2_REGULARIZATION:
    {
        Node *squared = pow(param, tensor(2.0f, 0));
        result = mul(tensor(0.5f * config.lambda, 0), squared);
        break;
    }
    case ELASTIC_NET:
    {
        Node *l1_term = mul(tensor(config.lambda * config.l1_ratio, 0),
                            pow(pow(param, tensor(2.0f, 0)), tensor(0.5f, 0)));
        Node *l2_term = mul(tensor(0.5f * config.lambda * (1.0f - config.l1_ratio), 0),
                            pow(param, tensor(2.0f, 0)));
        result = add(l1_term, l2_term);
        break;
    }
    default:
        result = tensor(0.0f, 0);
    }
    return result;
}

Node *apply_regularization_node(Node *param, RegularizerConfig config)
{
    if (!config.decay_only_weights)
    {
        return compute_regularization_node(param, config);
    }
    return tensor(0.0f, 0);
}

Node *apply_decoupled_weight_decay_node(Node *param, float weight_decay, float lr)
{
    if (weight_decay == 0.0f)
    {
        return param;
    }
    Node *scale = tensor(1.0f - lr * weight_decay, 0);
    return mul(param, scale);
}

Node *adjust_learning_rate_node(Node *lr, int epoch, LRSchedulerType scheduler_type,
                                float gamma, int step_size)
{
    Node *result = NULL;
    switch (scheduler_type)
    {
    case STEP_LR:
    {
        Node *power = tensor(floorf(epoch / step_size), 0);
        Node *gamma_node = tensor(gamma, 0);
        result = mul(lr, pow(gamma_node, power));
        break;
    }
    case EXPONENTIAL_LR:
    {
        Node *epoch_node = tensor(epoch, 0);
        Node *gamma_node = tensor(gamma, 0);
        result = mul(lr, pow(gamma_node, epoch_node));
        break;
    }
    case COSINE_LR:
    {
        Node *pi_node = tensor(M_PI, 0);
        Node *epoch_node = tensor(epoch, 0);
        Node *step_node = tensor(step_size, 0);
        Node *cos_term = mul(pi_node, div(epoch_node, step_node));
        Node *scale = mul(tensor(0.5f, 0), add(tensor(1.0f, 0), cos(cos_term)));
        result = mul(lr, scale);
        break;
    }
    default:
        result = lr;
    }
    return result;
}
