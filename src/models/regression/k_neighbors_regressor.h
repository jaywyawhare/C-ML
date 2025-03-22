#ifndef K_NEIGHBORS_REGRESSOR_H
#define K_NEIGHBORS_REGRESSOR_H

typedef struct
{
    float *X_train;
    float *y_train;
    int m;
    int n;
    int k;
} KNeighborsRegressor;

void knn_reg_fit(KNeighborsRegressor *model, float *X, float *y, int m, int n);
float knn_reg_predict(KNeighborsRegressor *model, float *x_input);
float knn_reg_score(KNeighborsRegressor *model, float *X, float *y, int m, int n);
void knn_reg_free_model(KNeighborsRegressor *model);

#endif // K_NEIGHBORS_REGRESSOR_H
