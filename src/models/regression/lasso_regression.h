#ifndef LASSO_REGRESSION_H
#define LASSO_REGRESSION_H

typedef struct
{
    float *coefficients;
    int num_features;
    float alpha;
    int max_iterations;
} LassoRegression;

void lasso_reg_fit(LassoRegression *model, float *X, float *y, int m, int n);
float lasso_reg_predict(LassoRegression *model, float *x_input);
float lasso_reg_score(LassoRegression *model, float *X, float *y, int m, int n);
void lasso_reg_get_params(LassoRegression *model);
void lasso_reg_set_params(LassoRegression *model, float *coefficients, float alpha);
void lasso_reg_free_model(LassoRegression *model);

#endif // LASSO_REGRESSION_H
