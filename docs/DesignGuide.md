# C-ML Codebase Design Guide

This document provides the conventions and best practices for contributing to the C-ML codebase. Adhering to these guidelines will ensure consistency, readability, and maintainability throughout the project.



## 1. General Coding Conventions

### 1.1 Indentation and Formatting
- **Indentation**: Use **4 spaces** for indentation (do not use tabs).
- **Braces**: Place opening braces `{` on the same line as the control statement (e.g., `if`, `for`, `while`).
  
  Example:
  ```c
  if (condition) {
      // Do something
  }
  ```
  
- **Blank Lines**: Use a single blank line to separate logical blocks of code. Avoid excessive blank lines.

### 1.2 Comments
- **Single-line comments**: Use `//` for brief explanations.
- **Multi-line comments**: Use `/* */` for longer descriptions.
- **Doxygen-style comments**: For documenting functions, structures, and files, use Doxygen-style comments (`/** */`).
  
  Example:
  ```c
  /**
   * @brief Calculates the forward pass of the pooling layer.
   * @param layer Pointer to the pooling layer structure.
   * @param input Pointer to the input data.
   * @param output Pointer to the output data.
   * @param input_size Size of the input data.
   * @return The number of output elements, or a negative error code.
   */
  int pooling_layer_forward(PoolingLayer *layer, const float *input, float *output, int input_size);
  ```

- **Avoid over-commenting**: Do not comment obvious code or trivial operations. Comments should focus on explaining "why" something is done, not "what" is done.



## 2. Variable Naming Conventions

### 2.1 General Naming Rules
- **Variable names**: Use **snake_case** for variables (e.g., `kernel_size`, `input_data`).
- **Descriptive names**: Choose meaningful names that indicate the purpose of the variable.
- **Short names**: Avoid one-character variable names except for loop counters (e.g., `i`, `j`).

### 2.2 Constants
- **Constant names**: Use **UPPERCASE_SNAKE_CASE** for constants (e.g., `MAX_BUFFER_SIZE`, `PI`).

### 2.3 Pointers
- **Pointer variables**: Prefix pointer variables with `p_` to indicate they are pointers (e.g., `p_layer`, `p_input`).



## 3. Function Naming Conventions

### 3.1 General Rules
- **Function names**: Use **snake_case** for function names (e.g., `initialize_layer`, `maxpooling_forward`).
- **Function naming format**: The function name should describe its purpose. Use the format:
  ```
  <module>_<action>_<specifics>
  ```
  Example: `pooling_layer_forward`, `activation_function_apply`.

### 3.2 Return Values
- **Success**: Return `0` to indicate success.
- **Errors**: Use negative values for error codes (e.g., `-1`, `-2`), and define them in an enumeration.

### 3.3 Documentation
- **Document each function** using Doxygen-style comments. Describe the function's purpose, parameters, and return values.
  


## 4. Layer Design Conventions

### 4.1 Structure Naming
- **Structure names**: Use **PascalCase** for structure names (e.g., `PoolingLayer`, `MaxPoolingLayer`).

### 4.2 Structure Fields
- **Structure field names**: Use **snake_case** for structure fields (e.g., `kernel_size`, `stride`).

### 4.3 Layer Functions
- Each layer should have the following functions:
  1. `layer_create`: Allocates and initializes the layer.
  2. `layer_forward`: Executes the forward pass.
  3. `layer_output_size`: Computes the output size.
  4. `layer_free`: Frees the allocated memory.

**Example**:
```c
PoolingLayer *pooling_layer_create(int kernel_size, int stride);
int pooling_layer_forward(PoolingLayer *layer, const float *input, float *output, int input_size);
int pooling_layer_output_size(int input_size, int kernel_size, int stride);
void pooling_layer_free(PoolingLayer *layer);
```



## 5. Debugging Conventions

### 5.1 Debug Logging
- **Enable/disable logs**: Use the `set_log_level(LOG_LEVEL_*)` macro to configure the global log level.
  ```c
  #include "include/logging.h"

  set_log_level(LOG_LEVEL_DEBUG);
  set_log_level(LOG_LEVEL_INFO);
  set_log_level(LOG_LEVEL_WARNING);
  set_log_level(LOG_LEVEL_ERROR);
  ```

- **Log appropriately**: Use the `#LOG_*` macros to conditionally log messages.
  ```c
  LOG_DEBUG("%s is a debug message.", message);
  LOG_INFO("Count is %d.", count);
  LOG_WARNING("Tensor Bloat is %d unreasonable.", bloat_factor);
  LOG_ERROR("NeuralNetwork is NULL.");
  ```

### 5.2 Error Messages
- **Error messages**: Use the `LOG_ERROR` macro. Include relevant parameter values.
  ```c
  LOG_ERROR("Invalid parameter (%d).", param);
  ```

### 5.3 Log Message Formatting
- Log messages are automatically formatted for you when you use the `LOG_*` macros.
```plaintext
Compiling and running test_logging...
Running logging tests...
2025-04-07 03:37:33 [DEBUG] test/Core/test_logging.c:18 main(): This is a debug message: 42
2025-04-07 03:37:33 [INFO] test/Core/test_logging.c:19 main(): This is an info message: hello
2025-04-07 03:37:33 [WARNING] test/Core/test_logging.c:20 main(): This is a warning message: 3.14
2025-04-07 03:37:33 [ERROR] test/Core/test_logging.c:21 main(): This is an error message: X
```

## 6. File Organization

### 6.1 Directory Structure
- **Source files**: Place `.c` files in the `src/` directory.
- **Header files**: Place `.h` files in the `include/` directory.
- **Subdirectories**: Use subdirectories for logical groupings, e.g., `Layers/` for layers, `core/` for core functionalities.

### 6.2 File Naming
- **File names**: Use **snake_case** for file names (e.g., `pooling.c`, `maxpooling.h`).



## 7. Error Handling

### 7.1 Error Codes
- **Error codes**: Define error codes using an enum in `src/core/error_codes.h`.
  ```c
  typedef enum {
      CM_SUCCESS = 0,
      CM_NULL_POINTER_ERROR = -1,
      CM_MEMORY_ALLOCATION_ERROR = -2,
      // Add other error codes as needed
  } CM_Error;
  ```

### 7.2 Error Propagation
- **Propagate errors**: Functions should return error codes (`CM_Error`), which can be checked by the caller.



## 8. Testing

### 8.1 Unit Tests
- **Test directory**: Place unit test files in the `tests/` directory.
- **Test coverage**: Ensure all layers and functions are tested.
  
### 8.2 Test Naming
- **File naming**: Use the format `<module>_test.c` for test files (e.g., `pooling_test.c`).



## 9. Code Review Checklist

Before submitting a pull request:
1. Code follows the naming and formatting conventions.
2. Functions are properly documented with Doxygen-style comments.
3. Logs use `LOG_DEBUG`, `LOG_INFO`, `LOG_WARNING` or `LOG_ERROR` appropriately.
4. Error handling is implemented and tested.
5. Unit tests are written and pass successfully.



## 10. Example Code

### 10.1 Pooling Layer Example
```c
#include "pooling.h"

int main() {
    // Create pooling layer
    PoolingLayer *layer = pooling_layer_create(2, 2);
    if (!layer) {
        LOG_ERROR("Failed to create pooling layer.\n");
        return CM_NULL_POINTER_ERROR;
    }

    // Perform forward pass
    float input[] = {1.0, 2.0, 3.0, 4.0};
    float output[2];
    int output_size = pooling_layer_forward(layer, input, output, 4);

    if (output_size < 0) {
        LOG_ERROR("Error during forward pass.\n");
        pooling_layer_free(layer);
        return output_size;
    }

    // Print output
    for (int i = 0; i < output_size; i++) {
        LOG_INFO("$2");Output[%d]: %f\n", i, output[i]);
    }

    // Free resources
    pooling_layer_free(layer);
    return CM_SUCCESS;
}
```



## 11. Memory Management

### 11.1 Custom Memory Allocation
- Use custom memory management functions:
  ```c
  void *cm_safe_malloc(size_t size, const char *file, int line);
  void cm_safe_free(void *ptr);
  ```

### 11.2 Memory Usage
- Always use `cm_safe_malloc` for memory allocation and `cm_safe_free` for deallocation.
- The `cm_safe_malloc` function should include file and line numbers for easier debugging.



## 12. Import Minimization

### 12.1 Avoid Unnecessary Imports
- Include only the necessary header files. Remove any unused imports to reduce compilation time and improve code clarity.

### 12.2 Minimize Library Imports
- When using external libraries, only import the specific components needed rather than the entire library. This reduces the size of the compiled code.



## 13. Comment Placement Guidelines

### 13.1 Implementation Comments in `.c` Files
- Use detailed comments in `.c` files to describe the implementation logic.
- Include explanations for complex computations, algorithms, or formulas.
- Focus on "how" the function works.

### 13.2 Interface Comments in `.h` Files
- Use concise comments in `.h` files to describe the function's purpose, parameters, and return values.
- Focus on "what" the function does and how other code will interact with it.
- Avoid including implementation details in `.h` files.

Example:
```c 
// .h file

/**
 * @brief Applies the GELU activation function.
 * @param input Pointer to the input array.
 * @param output Pointer to the output array.
 * @param size Number of elements in the input array.
 * @return CM_SUCCESS on success, or an error code on failure.
 */
int gelu_activation(const float *input, float *output, int size);
```

```c
// .c file

/**
 * @brief Applies the GELU activation function.
 *
 * The GELU function is defined as:
 * GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
 *
 * This implementation uses an approximation for efficiency.
 *
 * @param input Pointer to the input array.
 * @param output Pointer to the output array.
 * @param size Number of elements in the input array.
 * @return CM_SUCCESS on success, or an error code on failure.
 */

int gelu_activation(const float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        float x = input[i];
        output[i] = 0.5 * x * (1 + tanh(sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3))));
    }
    return CM_SUCCESS;
}
```
