#ifndef EXTRA_TREES_REGRESSOR_H
#define EXTRA_TREES_REGRESSOR_H

#include "decision_tree_regressor.h"

typedef struct
{
    DecisionTreeRegressor **trees;
    int num_trees;
    int max_depth;
    int min_samples_split;
} ExtraTreesRegressor;

void et_reg_init(ExtraTreesRegressor *model, int num_trees, int max_depth, int min_samples_split);
void et_reg_fit(ExtraTreesRegressor *model, float *X, float *y, int m, int n);
float et_reg_predict(ExtraTreesRegressor *model, float *x_input);
float et_reg_score(ExtraTreesRegressor *model, float *X, float *y, int m, int n);
void et_reg_free_model(ExtraTreesRegressor *model);

#endif // EXTRA_TREES_REGRESSOR_H
