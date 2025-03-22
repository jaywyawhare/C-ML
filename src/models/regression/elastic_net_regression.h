#ifndef ELASTIC_NET_REGRESSION_H
#define ELASTIC_NET_REGRESSION_H

typedef struct
{
    float *coefficients;
    int num_features;
    float alpha;
    float l1_ratio;
    int max_iterations;
} ElasticNetRegression;

void elastic_net_reg_fit(ElasticNetRegression *model, float *X, float *y, int m, int n);
float elastic_net_reg_predict(ElasticNetRegression *model, float *x_input);
float elastic_net_reg_score(ElasticNetRegression *model, float *X, float *y, int m, int n);
void elastic_net_reg_get_params(ElasticNetRegression *model);
void elastic_net_reg_set_params(ElasticNetRegression *model, float *coefficients, float alpha, float l1_ratio);
void elastic_net_reg_free_model(ElasticNetRegression *model);

#endif // ELASTIC_NET_REGRESSION_H
