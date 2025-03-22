#ifndef C_ML_ERROR_CODES_H
#define C_ML_ERROR_CODES_H

/**
 * @brief Defines the error codes for the C-ML library.
 */
typedef enum {
    CM_SUCCESS = 0,                  /**< Success */
    CM_NULL_POINTER_ERROR = -1,       /**< Null pointer error */
    CM_MEMORY_ALLOCATION_ERROR = -2,  /**< Memory allocation error */
    CM_INVALID_PARAMETER_ERROR = -3,   /**< Invalid parameter error */
    CM_INVALID_STRIDE_ERROR = -4,     /**< Invalid stride error */
    CM_INVALID_KERNEL_SIZE_ERROR = -5, /**< Invalid kernel size error */
    CM_INPUT_SIZE_SMALLER_THAN_KERNEL_ERROR = -6 /**< Input size smaller than kernel error */
} CM_Error;

#endif // C_ML_ERROR_CODES_H
