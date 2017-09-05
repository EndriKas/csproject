EXE 	= neuralnet
SRC		= $(wildcard $(SRC_DIR)/*.c)
OBJ 	= $(SRC:$(SRC_DIR)/%.c=$(OBJ_DIR)/%.o)
SRC_DIR = src
OBJ_DIR = obj
CC		= gcc
CFLAGS 	= -Wall -O2 -Iinclude
LDLIBS	= -lm -lgsl -lgslcblas



.PHONY: all clean

all: $(EXE)


$(EXE): $(OBJ)
	$(CC) $(CFLAGS) $^ $(LDLIBS) -o $@


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS)  -c $< -o $@


clean:
	rm -f $(EXE) $(OBJ)


