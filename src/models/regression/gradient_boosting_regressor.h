#ifndef GRADIENT_BOOSTING_REGRESSOR_H
#define GRADIENT_BOOSTING_REGRESSOR_H

#include "decision_tree_regressor.h"

typedef struct
{
    DecisionTreeRegressor **trees;
    int num_trees;
    float learning_rate;
    int max_depth;
    int min_samples_split;
} GradientBoostingRegressor;

void gb_reg_init(GradientBoostingRegressor *model, int num_trees, float learning_rate, int max_depth, int min_samples_split);
void gb_reg_fit(GradientBoostingRegressor *model, float *X, float *y, int m, int n);
float gb_reg_predict(GradientBoostingRegressor *model, float *x_input);
float gb_reg_score(GradientBoostingRegressor *model, float *X, float *y, int m, int n);
void gb_reg_free_model(GradientBoostingRegressor *model);

#endif // GRADIENT_BOOSTING_REGRESSOR_H
