################################################################################
# C-ML Machine Learning Library Makefile
################################################################################

# Compiler and flags
CC      := gcc
CFLAGS  := -g -Wall -MMD -Iinclude
LDFLAGS :=

# Project structure
SRC_DIR     := src
INCLUDE_DIR := include
OBJ_DIR     := obj
BIN_DIR     := bin
LIB_DIR     := lib
TEST_DIR    := test
TEST_BIN_DIR := test_bin
EXAMPLES_DIR := examples
EXAMPLES_BIN_DIR := examples_bin

# Library configuration
LIB_NAME    := c_ml
LIB_VERSION := 1.0.0
STATIC_LIB  := $(LIB_DIR)/lib$(LIB_NAME).a

# Find all source files
SRC_FILES   := $(shell find $(SRC_DIR) -name "*.c")
MAIN_FILE   := main.c
ALL_SRCS    := $(SRC_FILES) $(MAIN_FILE)

# Generate object file paths
OBJ_FILES   := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC_FILES))
MAIN_OBJ    := $(OBJ_DIR)/main.o
ALL_OBJS    := $(OBJ_FILES) $(MAIN_OBJ)
LIB_OBJS    := $(OBJ_FILES)

# Find all test files
TEST_FILES  := $(shell find $(TEST_DIR) -name "*.c")

# Find all example files
EXAMPLE_FILES := $(wildcard $(EXAMPLES_DIR)/*.c)
EXAMPLES      := $(patsubst $(EXAMPLES_DIR)/%.c,$(EXAMPLES_BIN_DIR)/%,$(EXAMPLE_FILES))

# Dependencies
DEPS := $(ALL_OBJS:.o=.d)

################################################################################
# Main targets
################################################################################

# Default target
.PHONY: all
all: $(BIN_DIR)/main

# Build the main executable
$(BIN_DIR)/main: $(ALL_OBJS)
	@mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@
	@echo "Built main executable: $@"

# Build the static library
.PHONY: lib
lib: $(STATIC_LIB)

$(STATIC_LIB): $(LIB_OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^
	@echo "Built static library: $@"

# Release build with optimizations
.PHONY: release
release: CFLAGS := -Wall -O2 -DNDEBUG -MMD
release: clean all
	@echo "Built release version"

# Debug build with sanitizers
.PHONY: debug
debug: CFLAGS += -fsanitize=address -fsanitize=undefined
debug: all
	@echo "Built debug version with sanitizers"

################################################################################
# Object file compilation
################################################################################

# Compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

# Compile main file
$(OBJ_DIR)/main.o: $(MAIN_FILE)
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

################################################################################
# Examples
################################################################################

.PHONY: examples
examples: $(STATIC_LIB) $(EXAMPLES)
	@echo "Built all examples"

$(EXAMPLES_BIN_DIR)/%: $(EXAMPLES_DIR)/%.c $(STATIC_LIB)
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $< -L$(LIB_DIR) -l$(LIB_NAME) -o $@
	@echo "Built example: $@"

# Debug version of examples with sanitizers
.PHONY: debug_examples
debug_examples: EXAMPLE_FLAGS := -DDEBUG_LOGGING -fsanitize=address -fsanitize=undefined
debug_examples: $(STATIC_LIB)
	@mkdir -p $(EXAMPLES_BIN_DIR)
	@for example_src in $(EXAMPLE_FILES); do \
		example_bin=$$(basename $$example_src .c); \
		echo "Compiling $$example_bin with debug flags..."; \
		$(CC) $(CFLAGS) $(EXAMPLE_FLAGS) $$example_src -L$(LIB_DIR) -l$(LIB_NAME) \
		-o $(EXAMPLES_BIN_DIR)/$$example_bin; \
	done
	@echo "Built all examples with debug flags"

# Neural network training example (special case)
.PHONY: nn_example
nn_example: $(STATIC_LIB)
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $(EXAMPLES_DIR)/nn_training_example.c -L$(LIB_DIR) -l$(LIB_NAME) -o $(EXAMPLES_BIN_DIR)/nn_training_example
	./$(EXAMPLES_BIN_DIR)/nn_training_example

# Debug version of neural network example
.PHONY: debug_nn_example
debug_nn_example: EXAMPLE_FLAGS := -DDEBUG_LOGGING -fsanitize=address -fsanitize=undefined
debug_nn_example: $(STATIC_LIB)
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $(EXAMPLE_FLAGS) $(EXAMPLES_DIR)/nn_training_example.c -L$(LIB_DIR) -l$(LIB_NAME) -o $(EXAMPLES_BIN_DIR)/nn_training_example
	./$(EXAMPLES_BIN_DIR)/nn_training_example

################################################################################
# Tests
################################################################################

.PHONY: test
test: $(STATIC_LIB)
	@mkdir -p $(TEST_BIN_DIR)
	@echo "Running tests..."
	@for test_src in $(TEST_FILES); do \
		test_bin=$$(basename $$test_src .c); \
		src_file=$$(echo $$test_src | sed 's|^test/|src/|; s|test_||'); \
		echo "\nCompiling and running $$test_bin..."; \
		$(CC) $(CFLAGS) $$test_src $$src_file -L$(LIB_DIR) -l$(LIB_NAME) \
		-o $(TEST_BIN_DIR)/$$test_bin -fsanitize=address -fsanitize=undefined && \
		ASAN_OPTIONS=allocator_may_return_null=1 ./$(TEST_BIN_DIR)/$$test_bin || exit 1; \
	done
	@echo "\nALL TESTS PASSED"

# Test logging system specifically
.PHONY: test_logging
test_logging: $(STATIC_LIB)
	@mkdir -p $(TEST_BIN_DIR)
	@echo "Running logging tests..."
	$(CC) $(CFLAGS) test/Core/test_logging.c src/Core/logging.c -o $(TEST_BIN_DIR)/test_logging
	./$(TEST_BIN_DIR)/test_logging

################################################################################
# Utility targets
################################################################################

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) $(TEST_BIN_DIR) $(EXAMPLES_BIN_DIR) $(LIB_DIR) a.out
	@echo "Cleaned all build artifacts"

# Include dependency files
-include $(DEPS)
