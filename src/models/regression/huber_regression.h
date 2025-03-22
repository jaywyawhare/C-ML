#ifndef HUBER_REGRESSION_H
#define HUBER_REGRESSION_H

typedef struct
{
    float *coefficients;
    int num_features;
    float alpha;
    float epsilon;
    int max_iterations;
} HuberRegression;

void huber_reg_fit(HuberRegression *model, float *X, float *y, int m, int n);
float huber_reg_predict(HuberRegression *model, float *x_input);
float huber_reg_score(HuberRegression *model, float *X, float *y, int m, int n);
void huber_reg_get_params(HuberRegression *model);
void huber_reg_set_params(HuberRegression *model, float *coefficients, float alpha, float epsilon);
void huber_reg_free_model(HuberRegression *model);

#endif // HUBER_REGRESSION_H
