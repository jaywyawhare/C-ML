typedef struct
{
    float *weights;
    float *biases;
    int input_size;
    int output_size;
} DenseLayer;

typedef struct
{
    float dropout_rate;
} DropoutLayer;

float elu(float x, float alpha);
float relu(float x);
float sigmoid(float x);
float tanH(float x);
float *softmax(float *x, int size);
float linear(float x);
float leakyRelu(float x);

void initializeDense(DenseLayer *layer, int input_size, int output_size);
void forwardDense(DenseLayer *layer, float *input, float *output);
void backwardDense(DenseLayer *layer, float *input, float *output, float *d_output, float *d_input, float *d_weights, float *d_biases);
void updateDense(DenseLayer *layer, float *d_weights, float *d_biases, float learning_rate);
void freeDense(DenseLayer *layer);

void initializeDropout(DropoutLayer *layer, float dropout_rate);
void forwardDropout(DropoutLayer *layer, float *input, float *output, int size);
void backwardDropout(DropoutLayer *layer, float *input, float *output, float *d_output, float *d_input, int size);

void freeSoftmax(float *output);

float Adam(float x, float y, float lr, float *w, float *b, float *v_w, float *v_b, float *s_w, float *s_b, float beta1, float beta2, float epsilon);
float RMSprop(float x, float y, float lr, float *w, float *b, float *cache_w, float *cache_b, float epsilon, float beta1, float beta2);
float SGD(float x, float y, float lr, float *w, float *b);

float l1_l2(float *w, float *dw, float *w_l1, float *dw_l1, float *w_l2, float *dw_l2, float l1, float l2, int n);
float l1(float x, float y, float lr, float w, float b, float v_w, float v_b, float s_w, float s_b, float beta1, float beta2, float epsilon);
float l2(float x, float y, float lr, float w, float b, float v_w, float v_b, float s_w, float s_b, float beta1, float beta2, float epsilon);

float rootMeanSquaredError(float *y, float *yHat, int n);
float reduceMean(float *loss, int size);
float meanSquaredError(float *y, float *yHat, int n);
float meanAbsolutePercentageError(float *y, float *yHat, int n);
float meanAbsoluteError(float *y, float *yHat, int n);
float focalLoss(float *y, float *yHat, int n, float gamma);
float binaryCrossEntropyLoss(float *yHat, float *y, int size);