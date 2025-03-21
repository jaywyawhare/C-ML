# C-ML Codebase Design Guide

This document provides the conventions and best practices for contributing to the C-ML codebase. Adhering to these guidelines will ensure consistency, readability, and maintainability throughout the project.

---

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

---

## 2. Variable Naming Conventions

### 2.1 General Naming Rules
- **Variable names**: Use **snake_case** for variables (e.g., `kernel_size`, `input_data`).
- **Descriptive names**: Choose meaningful names that indicate the purpose of the variable.
- **Short names**: Avoid one-character variable names except for loop counters (e.g., `i`, `j`).

### 2.2 Constants
- **Constant names**: Use **UPPERCASE_SNAKE_CASE** for constants (e.g., `MAX_BUFFER_SIZE`, `PI`).

### 2.3 Pointers
- **Pointer variables**: Prefix pointer variables with `p_` to indicate they are pointers (e.g., `p_layer`, `p_input`).

---

## 3. Function Naming Conventions

### 3.1 General Rules
- **Function names**: Use **snake_case** for function names (e.g., `initialize_layer`, `maxpooling_forward`).
- **Function naming format**: The function name should describe its purpose. Use the format:
  ```
  <module>_<action>_<specifics>
  ```
  Example: `polling_layer_forward`, `activation_function_apply`.

### 3.2 Return Values
- **Success**: Return `0` to indicate success.
- **Errors**: Use negative values for error codes (e.g., `-1`, `-2`), and define them in an enumeration.

### 3.3 Documentation
- **Document each function** using Doxygen-style comments. Describe the function's purpose, parameters, and return values.
  
---

## 4. Layer Design Conventions

### 4.1 Structure Naming
- **Structure names**: Use **PascalCase** for structure names (e.g., `PollingLayer`, `MaxPoolingLayer`).

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
PollingLayer *polling_layer_create(int kernel_size, int stride);
int polling_layer_forward(PollingLayer *layer, const float *input, float *output, int input_size);
int polling_layer_output_size(int input_size, int kernel_size, int stride);
void polling_layer_free(PollingLayer *layer);
```

---

## 5. Debugging Conventions

### 5.1 Debug Logging
- **Enable/disable logs**: Use the `DEBUG_LOGGING` macro to toggle debug logs.
  ```c
  #define DEBUG_LOGGING 1  // Set to 0 to disable debug logs
  ```

- **Wrap debug logs**: Use `#if DEBUG_LOGGING` to conditionally compile debug messages.
  ```c
  #if DEBUG_LOGGING
  printf("[function_name] Debug: %s\n", message);
  #endif
  ```

### 5.2 Error Messages
- **Error messages**: Include the function name and relevant parameter values.
  ```c
  fprintf(stderr, "[function_name] Error: Invalid parameter (%d).\n", param);
  ```

---

## 6. File Organization

### 6.1 Directory Structure
- **Source files**: Place `.c` files in the `src/` directory.
- **Header files**: Place `.h` files in the `include/` directory.
- **Subdirectories**: Use subdirectories for logical groupings, e.g., `Layers/` for layers, `core/` for core functionalities.

### 6.2 File Naming
- **File names**: Use **snake_case** for file names (e.g., `polling.c`, `maxpooling.h`).

---

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

---

## 8. Testing

### 8.1 Unit Tests
- **Test directory**: Place unit test files in the `tests/` directory.
- **Test coverage**: Ensure all layers and functions are tested.
  
### 8.2 Test Naming
- **File naming**: Use the format `<module>_test.c` for test files (e.g., `polling_test.c`).

---

## 9. Code Review Checklist

Before submitting a pull request:
1. Code follows the naming and formatting conventions.
2. Functions are properly documented with Doxygen-style comments.
3. Debug logs are wrapped in `#if DEBUG_LOGGING` blocks.
4. Error handling is implemented and tested.
5. Unit tests are written and pass successfully.

---

## 10. Example Code

### 10.1 Polling Layer Example
```c
#include "polling.h"

int main() {
    // Create polling layer
    PollingLayer *layer = polling_layer_create(2, 2);
    if (!layer) {
        fprintf(stderr, "Failed to create polling layer.\n");
        return CM_NULL_POINTER_ERROR;
    }

    // Perform forward pass
    float input[] = {1.0, 2.0, 3.0, 4.0};
    float output[2];
    int output_size = polling_layer_forward(layer, input, output, 4);

    if (output_size < 0) {
        fprintf(stderr, "Error during forward pass.\n");
        polling_layer_free(layer);
        return output_size;
    }

    // Print output
    for (int i = 0; i < output_size; i++) {
        printf("Output[%d]: %f\n", i, output[i]);
    }

    // Free resources
    polling_layer_free(layer);
    return CM_SUCCESS;
}
```

---

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

---

## 12. Import Minimization

### 12.1 Avoid Unnecessary Imports
- Include only the necessary header files. Remove any unused imports to reduce compilation time and improve code clarity.

### 12.2 Minimize Library Imports
- When using external libraries, only import the specific components needed rather than the entire library. This reduces the size of the compiled code.

---

By adhering to these guidelines, contributors will maintain a clean, organized, and efficient C-ML codebase that is easy to understand, extend, and debug.