CC = gcc
CFLAGS = -O3 -march=native -mavx2 -mavxvnni -mfma -fopenmp -pthread -Wall -Wextra -Iinclude -D_GNU_SOURCE
LDFLAGS = -fopenmp -pthread -ldl -lm

SRCS = src/hardware.c src/simd_vnni.c src/cuda_backend.c src/nvme_streamer.c src/sparsity.c src/priority.c src/pipeline_stages.c src/tokenizer.c src/engine.c
OBJS = $(SRCS:.c=.o)

all: noel aether benchmark generate_weights benchmark_cpu_offload test_engine libnoel.so

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

libnoel.so: $(SRCS)
	$(CC) -fPIC -shared $(CFLAGS) $(SRCS) $(LDFLAGS) -o libnoel.so

noel: $(OBJS) src/main.o
	$(CC) $(OBJS) src/main.o $(LDFLAGS) -o noel

aether: noel
	ln -sf noel aether

benchmark: $(OBJS) tools/benchmark_tiers.o
	$(CC) $(OBJS) tools/benchmark_tiers.o $(LDFLAGS) -o benchmark

generate_weights: tools/generate_weights.o
	$(CC) tools/generate_weights.o $(LDFLAGS) -o generate_weights

benchmark_cpu_offload: $(OBJS) tools/benchmark_cpu_offload.o
	$(CC) $(OBJS) tools/benchmark_cpu_offload.o $(LDFLAGS) -o benchmark_cpu_offload

test_engine: $(OBJS) tests/test_engine.o
	$(CC) $(OBJS) tests/test_engine.o $(LDFLAGS) -o test_engine

test: test_engine
	./test_engine

clean:
	rm -f src/*.o tools/*.o tests/*.o noel aether benchmark generate_weights benchmark_cpu_offload test_engine libnoel.so *.so

.PHONY: all test clean
