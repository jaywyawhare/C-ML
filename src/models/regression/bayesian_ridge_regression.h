#ifndef BAYESIAN_RIDGE_REGRESSION_H
#define BAYESIAN_RIDGE_REGRESSION_H

typedef struct
{
    float *coefficients;
    int num_features;
    float alpha;
    float lambda;
} BayesianRidgeRegression;

void bayesian_ridge_reg_fit(BayesianRidgeRegression *model, float *X, float *y, int m, int n);
float bayesian_ridge_reg_predict(BayesianRidgeRegression *model, float *x_input);
void bayesian_ridge_reg_get_params(BayesianRidgeRegression *model);
void bayesian_ridge_reg_set_params(BayesianRidgeRegression *model, float *coefficients, float alpha, float lambda);
void bayesian_ridge_reg_free_model(BayesianRidgeRegression *model);

#endif // BAYESIAN_RIDGE_REGRESSION_H
