CC      = gcc
CFLAGS  = -g -Wall -MMD -lm
SRC     = src
OBJ     = obj
BINDIR  = bin
LIB     = libmylib.a
TEST_BIN_DIR = test_bin

SRCS    = $(wildcard $(SRC)/**/*.c) main.c
OBJS    = $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(filter $(SRC)%,$(SRCS))) \
          $(patsubst %.c, $(OBJ)/%.o, $(filter main.c,$(SRCS)))
LIB_OBJS = $(filter-out $(OBJ)/main.o, $(OBJS))
DEPS    = $(OBJS:.o=.d)

TEST_SRCS = $(wildcard test/**/*.c)

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

clean:
	rm -rf $(BINDIR) $(OBJ) $(TEST_BIN_DIR) $(LIB) a.out

-include $(DEPS)
