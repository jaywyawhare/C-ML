float reduceMean(float *loss) {
    float sum = 0;
    for (int i = 0; i < sizeof(loss); i++) {
        sum += loss[i];
    }
    return sum / sizeof(loss);
}