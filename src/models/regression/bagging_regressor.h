#ifndef BAGGING_REGRESSOR_H
#define BAGGING_REGRESSOR_H

#include "decision_tree_regressor.h"

typedef struct
{
    DecisionTreeRegressor **base_estimators;
    int num_estimators;
    int max_depth;
    int min_samples_split;
} BaggingRegressor;

void bagging_reg_init(BaggingRegressor *model, int num_estimators, int max_depth, int min_samples_split);
void bagging_reg_fit(BaggingRegressor *model, float *X, float *y, int m, int n);
float bagging_reg_predict(BaggingRegressor *model, float *x_input);
float bagging_reg_score(BaggingRegressor *model, float *X, float *y, int m, int n);
void bagging_reg_free_model(BaggingRegressor *model);

#endif // BAGGING_REGRESSOR_H
