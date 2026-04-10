#include "cml.h"
#include <stdio.h>

int main(void) {
    if (cml_init() != 0) {
        printf("Error: cml_init failed\n");
        return 1;
    }
    cml_seed(42);

    const int input_size = 8;
    const int output_size = 1;
    const int num_samples = 100;

    DeviceType device = cml_get_default_device();
    DType dtype = cml_get_default_dtype();

    Sequential* model = nn_sequential();
    model = sequential_add_chain(model, (Module*)nn_linear(input_size, 32, dtype, device, true));
    model = sequential_add_chain(model, (Module*)nn_relu(false));
    model = sequential_add_chain(model, (Module*)nn_linear(32, 16, dtype, device, true));
    model = sequential_add_chain(model, (Module*)nn_relu(false));
    model = sequential_add_chain(model, (Module*)nn_linear(16, output_size, dtype, device, true));
    model = sequential_add_chain(model, (Module*)nn_sigmoid());

    Optimizer* optimizer = optim_adam_for_model((Module*)model, 0.01f, 0.0f, 0.9f, 0.999f, 1e-8f);
    if (!optimizer) {
        printf("Error: failed to create optimizer\n");
        cml_cleanup();
        return 1;
    }

    Dataset* full_dataset = dataset_random_classification(num_samples, input_size, output_size);
    if (!full_dataset) {
        printf("Error: failed to create dataset\n");
        cml_cleanup();
        return 1;
    }

    Dataset* train_dataset = NULL;
    Dataset* val_dataset = NULL;
    Dataset* test_dataset = NULL;
    if (dataset_split_three(full_dataset, 0.7f, 0.15f, &train_dataset, &val_dataset, &test_dataset) != 0) {
        printf("Error: failed to split dataset\n");
        cml_cleanup();
        return 1;
    }

    DataLoader* train_loader = dataloader_create(train_dataset, 16, false);
    DataLoader* val_loader = dataloader_create(val_dataset, 16, false);
    if (!train_loader || !val_loader) {
        printf("Error: failed to create dataloaders\n");
        if (train_loader) dataloader_free(train_loader);
        if (val_loader) dataloader_free(val_loader);
        dataset_free(train_dataset);
        dataset_free(val_dataset);
        dataset_free(test_dataset);
        cml_cleanup();
        return 1;
    }

    TrainingConfig cfg;
    training_config_default(&cfg);
    cfg.epochs = 3;
    cfg.verbose = false;
    cfg.use_progress_bar = false;

    if (cml_train_with_validation((Module*)model, train_loader, val_loader, optimizer,
                                  tensor_mse_loss, &cfg) != 0) {
        printf("Error: training failed\n");
        dataloader_free(train_loader);
        dataloader_free(val_loader);
        dataset_free(train_dataset);
        dataset_free(val_dataset);
        dataset_free(test_dataset);
        cml_cleanup();
        return 1;
    }

    if (training_metrics_evaluate_dataset((Module*)model, test_dataset, tensor_mse_loss, false) != 0) {
        printf("Error: test evaluation failed\n");
        dataloader_free(train_loader);
        dataloader_free(val_loader);
        dataset_free(train_dataset);
        dataset_free(val_dataset);
        dataset_free(test_dataset);
        cml_cleanup();
        return 1;
    }

    dataloader_free(train_loader);
    dataloader_free(val_loader);
    dataset_free(train_dataset);
    dataset_free(val_dataset);
    dataset_free(test_dataset);

    cml_cleanup();
    return 0;
}
