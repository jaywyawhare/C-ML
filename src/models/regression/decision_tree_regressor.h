#ifndef DECISION_TREE_REGRESSOR_H
#define DECISION_TREE_REGRESSOR_H

typedef struct DecisionTreeNode
{
    int feature_index;
    float threshold;
    float value;
    struct DecisionTreeNode *left;
    struct DecisionTreeNode *right;
} DecisionTreeNode;

typedef struct
{
    DecisionTreeNode *root;
    int max_depth;
    int min_samples_split;
} DecisionTreeRegressor;

void dt_reg_fit(DecisionTreeRegressor *model, float *X, float *y, int m, int n);
float dt_reg_predict(DecisionTreeRegressor *model, float *x_input);
void dt_reg_free_model(DecisionTreeRegressor *model);

#endif // DECISION_TREE_REGRESSOR_H
