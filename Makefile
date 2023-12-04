CC      = gcc
CFLAGS  = -g -Wall
SRC     = src
OBJ     = obj
SRCS    = $(wildcard $(SRC)/*.c) $(wildcard $(SRC)/Activations/*.c) $(wildcard $(SRC)/Layers/*.c) $(wildcard $(SRC)/Optimizers/*.c) $(wildcard $(SRC)/Regularizers/*.c)
OBJS    = $(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))

BINDIR  = bin
BIN     = $(BINDIR)/main
SUBMITNAME = code.zip
ZIP     = zip

all: $(BIN)

release: CFLAGS = -Wall -O2 -DNDEBUG
release: clean $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $@ -lm

$(OBJ)/%.o: $(SRC)/%.c
	mkdir -p $(@D)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	$(RM) -r $(BINDIR)/* $(OBJ)/*
