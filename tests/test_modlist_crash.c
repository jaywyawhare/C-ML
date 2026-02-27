#include "cml.h"
#include <stdio.h>

int main(void) {
    setbuf(stdout, NULL); // unbuffered
    printf("step 1: init\n");
    cml_init();
    printf("step 2: creating model\n");

    Sequential* model = cml_nn_sequential();
    cml_nn_sequential_add(model, (Module*)cml_nn_linear(2, 4, DTYPE_FLOAT32, DEVICE_CPU, true));
    printf("step 3: model created\n");

    printf("step 4: creating ModuleList\n");
    ModuleList* list = cml_nn_module_list();
    printf("step 5: list=%p\n", (void*)list);

    printf("step 6: appending\n");
    module_list_append(list, (Module*)model);
    printf("step 7: length=%d\n", module_list_length(list));

    printf("step 8: cleanup\n");
    cml_cleanup();
    printf("step 9: done\n");
    return 0;
}
