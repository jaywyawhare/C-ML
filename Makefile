# C-ML Library Makefile
# Builds the C-ML machine learning library with high-level API

# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -O2 -g
LDFLAGS = -lm

# Directories
INCLUDE_DIR = include
SRC_DIR = src
BUILD_DIR = build
TEST_DIR = test

# Source files
CORE_SOURCES = $(SRC_DIR)/Core/logging.c \
               $(SRC_DIR)/Core/memory_management.c \
               $(SRC_DIR)/Core/memory_pools.c \
               $(SRC_DIR)/Core/dataset.c \
               $(SRC_DIR)/Core/augmentation.c \
               $(SRC_DIR)/Core/profiling.c

TENSOR_SOURCES = $(SRC_DIR)/tensor.c \
                 $(SRC_DIR)/tensor_views.c \
                 $(SRC_DIR)/tensor_manipulation.c
AUTOGRAD_SOURCES = $(SRC_DIR)/autograd/autograd.c \
                   $(SRC_DIR)/autograd/backward_ops.c \
                   $(SRC_DIR)/autograd/forward_ops.c \
                   $(SRC_DIR)/autograd/loss_functions.c \
                   $(SRC_DIR)/autograd/checkpointing.c
NN_SOURCES = $(SRC_DIR)/nn.c \
            $(SRC_DIR)/nn/layers/linear.c \
            $(SRC_DIR)/nn/layers/activations.c \
            $(SRC_DIR)/nn/layers/dropout.c \
            $(SRC_DIR)/nn/layers/conv2d.c \
            $(SRC_DIR)/nn/layers/batchnorm2d.c \
            $(SRC_DIR)/nn/layers/layernorm.c \
            $(SRC_DIR)/nn/layers/pooling.c \
            $(SRC_DIR)/nn/layers/sequential.c
OPTIM_SOURCES = $(SRC_DIR)/optim.c

# Main library source
LIBRARY_SOURCES = $(SRC_DIR)/cml.c \
                  $(CORE_SOURCES) \
                  $(TENSOR_SOURCES) \
                  $(AUTOGRAD_SOURCES) \
                  $(NN_SOURCES) \
                  $(OPTIM_SOURCES)

# Object files
LIBRARY_OBJECTS = $(LIBRARY_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

# Main executable
MAIN_SOURCE = main.c
MAIN_OBJECT = $(BUILD_DIR)/main.o

# Test sources
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%.o)
TEST_EXECUTABLES = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BUILD_DIR)/test_%)

# Example sources
EXAMPLES_DIR = examples
EXAMPLE_SOURCES = $(EXAMPLES_DIR)/autograd_example.c \
                  $(EXAMPLES_DIR)/training_loop_example.c \
                  $(EXAMPLES_DIR)/opcheck.c \
                  $(EXAMPLES_DIR)/bench_gemm.c
EXAMPLE_EXECUTABLES = $(EXAMPLE_SOURCES:$(EXAMPLES_DIR)/%.c=$(BUILD_DIR)/examples/%)

# Default target
all: $(BUILD_DIR)/main $(EXAMPLE_EXECUTABLES)

# Create build directories
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)
	mkdir -p $(BUILD_DIR)/Core
	mkdir -p $(BUILD_DIR)/tensor
	mkdir -p $(BUILD_DIR)/autograd
	mkdir -p $(BUILD_DIR)/nn
	mkdir -p $(BUILD_DIR)/nn/layers
	mkdir -p $(BUILD_DIR)/optim
	mkdir -p $(BUILD_DIR)/examples

# Main executable
$(BUILD_DIR)/main: $(BUILD_DIR) $(MAIN_OBJECT) $(LIBRARY_OBJECTS)
	$(CC) $(MAIN_OBJECT) $(LIBRARY_OBJECTS) -o $@ $(LDFLAGS)

# Main object file
$(MAIN_OBJECT): $(MAIN_SOURCE) | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Library object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Tests
test: $(BUILD_DIR) $(TEST_EXECUTABLES)
	@echo "Running tests..."
	@for test in $(TEST_EXECUTABLES); do \
		echo "Running $$test..."; \
		$$test; \
	done

# Test executables
$(BUILD_DIR)/test_%: $(BUILD_DIR)/test_%.o $(LIBRARY_OBJECTS)
	$(CC) $< $(LIBRARY_OBJECTS) -o $@ $(LDFLAGS)

# Test object files
$(BUILD_DIR)/test_%.o: $(TEST_DIR)/%.c | $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Install (optional)
install: all
	@echo "Installing C-ML library..."
	@echo "Note: This is a placeholder. Implement proper installation as needed."

# Debug build
debug: CFLAGS += -DDEBUG -O0 -g3
debug: all

# Release build
release: CFLAGS += -DNDEBUG -O3
release: all

# Static analysis
analyze: CFLAGS += -fanalyzer
analyze: clean all

# Help
help:
	@echo "C-ML Library Makefile"
	@echo "====================="
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build the main executable (default)"
	@echo "  test       - Build and run tests"
	@echo "  debug      - Build with debug flags"
	@echo "  release    - Build with release flags"
	@echo "  analyze    - Build with static analysis"
	@echo "  clean      - Remove build artifacts"
	@echo "  install    - Install the library (placeholder)"
	@echo ""
	@echo "Examples: built to $(BUILD_DIR)/examples/"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Variables:"
	@echo "  CC         - C compiler (default: gcc)"
	@echo "  CFLAGS     - C compiler flags"
	@echo "  LDFLAGS    - Linker flags"
	@echo "  INCLUDE_DIR - Include directory (default: include)"
	@echo "  SRC_DIR    - Source directory (default: src)"
	@echo "  BUILD_DIR  - Build directory (default: build)"

# Phony targets
.PHONY: all test clean install debug release analyze help

# Dependencies
-include $(LIBRARY_OBJECTS:.o=.d)
-include $(MAIN_OBJECT:.o=.d)
-include $(TEST_OBJECTS:.o=.d)

# Examples build rules
$(BUILD_DIR)/examples/%: $(EXAMPLES_DIR)/%.c $(LIBRARY_OBJECTS) | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) $< $(LIBRARY_OBJECTS) -o $@ $(LDFLAGS)
