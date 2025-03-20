CC      = gcc
CFLAGS  = -g -Wall -lm
SRC     = src
OBJ     = obj
BINDIR  = bin
BIN     = $(BINDIR)/main

TEST_BIN_DIR = test_bin

SRCS    = $(wildcard $(SRC)/**/*.c) main.c
OBJS    = $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))

TEST_SRCS = $(wildcard test/**/*.c)

all: $(BIN)

release: CFLAGS = -Wall -O2 -DNDEBUG
release: clean $(BIN)

.PHONY: test
test:
	@echo "Running tests..."

	mkdir -p $(BINDIR)
	mkdir -p $(TEST_BIN_DIR)

	$(foreach test_src, $(TEST_SRCS), \
		test_bin=$$(basename $(notdir $(test_src))); \
		test_folder=$$(dirname $(test_src)); \
		src_file=$$(echo $(test_src) | sed 's|test/|src/|;s|test_||'); \
		echo ""; \
		$(CC) $(CFLAGS) $$test_folder/$$test_bin $$src_file -lm -o $(TEST_BIN_DIR)/$$test_bin && ./$(TEST_BIN_DIR)/$$test_bin || exit 1; \
	)

	@echo "\nALL TESTS PASSED"


$(BIN): $(OBJS)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(OBJS) -o $@ -lm

$(OBJ)/%.o: $(SRC)/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/main.o: main.c
	mkdir -p $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BINDIR) $(OBJ) $(TEST_BIN_DIR) a.out
