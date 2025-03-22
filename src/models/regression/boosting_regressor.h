#ifndef BOOSTING_REGRESSOR_H
#define BOOSTING_REGRESSOR_H

typedef struct
{
    float *weights;
    float bias;
    int num_features;
    int num_trees;
    float learning_rate;
    float reg_lambda;
} BoostingRegressor;

void boosting_reg_init(BoostingRegressor *model, int num_features, int num_trees, float learning_rate, float reg_lambda);
void boosting_reg_fit(BoostingRegressor *model, float *X, float *y, int m, int n);
void boosting_reg_fit_single_model(BoostingRegressor *model, float *X, float *residuals, int m, int n);
float boosting_reg_predict(BoostingRegressor *model, float *x_input);
void boosting_reg_free_model(BoostingRegressor *model);
void boosting_reg_save_model(BoostingRegressor *model, const char *filename);
void boosting_reg_load_model(BoostingRegressor *model, const char *filename);

#endif // BOOSTING_REGRESSOR_H
