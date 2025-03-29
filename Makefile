CC      = gcc
CFLAGS  = -g -Wall -MMD -lm -Iinclude
SRC     = src
OBJ     = obj
BINDIR  = bin
LIB     = libmylib.a
TEST_BIN_DIR = test_bin
EXAMPLES_DIR = examples
EXAMPLES_BIN_DIR = examples_bin

SRCS    = $(wildcard $(SRC)/**/*.c) main.c
OBJS    = $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(filter $(SRC)%,$(SRCS))) \
          $(patsubst %.c, $(OBJ)/%.o, $(filter main.c,$(SRCS)))
LIB_OBJS = $(filter-out $(OBJ)/main.o, $(OBJS))
DEPS    = $(OBJS:.o=.d)

TEST_SRCS = $(wildcard test/**/*.c)
EXAMPLES_SRCS = $(wildcard $(EXAMPLES_DIR)/*.c)
EXAMPLES = $(patsubst $(EXAMPLES_DIR)/%.c, $(EXAMPLES_BIN_DIR)/%, $(EXAMPLES_SRCS))

all: $(BINDIR)/main

release: CFLAGS = -Wall -O2 -DNDEBUG -MMD
release: clean all

$(BINDIR)/main: $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(OBJS) -o $@ -lm

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/main.o: main.c
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIB): $(LIB_OBJS)
	ar rcs $@ $^

.PHONY: examples
examples: $(LIB) $(EXAMPLES)

$(EXAMPLES_BIN_DIR)/%: $(EXAMPLES_DIR)/%.c
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $< -L. -lmylib -lm -o $@

.PHONY: test
test: $(LIB)
	@mkdir -p $(TEST_BIN_DIR)
	@echo "Running tests..."
	@for test_src in $(TEST_SRCS); do \
		test_bin=$$(basename $$test_src .c); \
		src_file=$$(echo $$test_src | sed 's|^test/|src/|; s|test_||'); \
		echo "\nCompiling and running $$test_bin..."; \
		$(CC) $(CFLAGS) $$test_src $$src_file -L. -lmylib -lm -o $(TEST_BIN_DIR)/$$test_bin && ./$(TEST_BIN_DIR)/$$test_bin || exit 1; \
	done
	@echo "\nALL TESTS PASSED"

.PHONY: nn_example
nn_example: $(LIB)
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $(EXAMPLES_DIR)/nn_training_example.c -L. -lmylib -lm -o $(EXAMPLES_BIN_DIR)/nn_training_example
	./$(EXAMPLES_BIN_DIR)/nn_training_example

clean:
	rm -rf $(BINDIR) $(OBJ) $(TEST_BIN_DIR) $(EXAMPLES_BIN_DIR) $(LIB) a.out

-include $(DEPS)
