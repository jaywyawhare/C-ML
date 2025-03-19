CC      = gcc
CFLAGS  = -g -Wall -lm
SRC     = src
OBJ     = obj
SRCS    = $(wildcard $(SRC)/*.c) \
          $(wildcard $(SRC)/Activations/*.c) \
          $(wildcard $(SRC)/Layers/*.c) \
          $(wildcard $(SRC)/Loss_Functions/*.c) \
          $(wildcard $(SRC)/Optimizers/*.c) \
          $(wildcard $(SRC)/Preprocessing/*.c) \
          $(wildcard $(SRC)/Regularizers/*.c) \
          main.c
OBJS    = $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))
OBJS    := $(OBJS)  

BINDIR  = bin
BIN     = $(BINDIR)/main

all: $(BIN)

release: CFLAGS = -Wall -O2 -DNDEBUG
release: clean $(BIN)

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
	rm -rf $(BINDIR) $(OBJ)
