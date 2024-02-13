CC=gcc
CFLAGS=-ggdb -std=c11 -O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wshadow -march=native -mtune=native -fno-omit-frame-pointer -DCL_TARGET_OPENCL_VERSION=220

LDFLAGS=-lOpenCL -lGL -lSDL2 -lm -L/home/mael/projects/glew-2.1.0/lib -lGLEW -lEGL -Wl,-rpath=/home/mael/projects/glew-2.1.0/lib

BUILD_DIR=./build
HEADERS=$(wildcard *.h)
OBJS=$(patsubst %.c,$(BUILD_DIR)/%.o,$(wildcard *.c))

TARGET=$(BUILD_DIR)/program

.PHONY: all, clean, run

all: $(TARGET)

run: $(TARGET)
	export LD_LIBRARY_PATH=$(LD_LIBRARY_PATH) && $(BUILD_DIR)/program

clean:
	rm build/*

# compile object files
$(BUILD_DIR)/%.o: %.c %.h
	$(CC) $(CFLAGS) -c $< -o $@

# compile main file
$(BUILD_DIR)/main.o: main.c *.h
	$(CC) $(CFLAGS) -c $< -o $@

# link program
$(TARGET): $(OBJS) $(HEADERS)
	$(CC) $(OBJS) -o $(TARGET) $(LDFLAGS)

