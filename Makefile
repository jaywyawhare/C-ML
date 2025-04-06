CC      = gcc
CFLAGS  = -g -Wall -MMD -Iinclude
SRC     = src
OBJ     = obj
BINDIR  = bin
LIBDIR  = lib
LIB_NAME = c_ml
LIB     = lib$(LIB_NAME).a
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
lib: $(LIBDIR)/$(LIB)
release: CFLAGS = -Wall -O2 -DNDEBUG -MMD
release: clean all

debug: CFLAGS += -DDEBUG_LOGGING -fsanitize=address -fsanitize=undefined
debug: all

$(BINDIR)/main: $(OBJS)
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(OBJS) -o $@ 

$(OBJ)/%.o: $(SRC)/%.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/main.o: main.c
	@mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBDIR)/$(LIB): $(LIB_OBJS)
	@mkdir -p $(LIBDIR)
	ar rcs $@ $^

.PHONY: examples
examples: $(LIBDIR)/$(LIB) $(EXAMPLES)

$(EXAMPLES_BIN_DIR)/%: $(EXAMPLES_DIR)/%.c
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $< -L$(LIBDIR) -l$(LIB_NAME) -o $@

.PHONY: test
test: $(LIBDIR)/$(LIB)
	@mkdir -p $(TEST_BIN_DIR)
	@echo "Running tests..."
	@for test_src in $(TEST_SRCS); do \
		test_bin=$$(basename $$test_src .c); \
		src_file=$$(echo $$test_src | sed 's|^test/|src/|; s|test_||'); \
		echo "\nCompiling and running $$test_bin..."; \
		$(CC) $(CFLAGS) $$test_src $$src_file -L$(LIBDIR) -l$(LIB_NAME) \
		-o $(TEST_BIN_DIR)/$$test_bin -fsanitize=address -fsanitize=undefined && \
		ASAN_OPTIONS=allocator_may_return_null=1 ./$(TEST_BIN_DIR)/$$test_bin || exit 1; \
	done
	@echo "\nALL TESTS PASSED"

.PHONY: nn_example
nn_example: $(LIBDIR)/$(LIB)
	@mkdir -p $(EXAMPLES_BIN_DIR)
	$(CC) $(CFLAGS) $(EXAMPLES_DIR)/nn_training_example.c -L$(LIBDIR) -l$(LIB_NAME) -o $(EXAMPLES_BIN_DIR)/nn_training_example -fsanitize=address -fsanitize=undefined
	./$(EXAMPLES_BIN_DIR)/nn_training_example

clean:
	rm -rf $(BINDIR) $(OBJ) $(TEST_BIN_DIR) $(EXAMPLES_BIN_DIR) $(LIBDIR) a.out

-include $(DEPS)
