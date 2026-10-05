NVCC = nvcc

NVCCFLAGS = -O3 -std=c++11 -arch=sm_89

SRC_DIR = src
EXEC = radix_sort.exe

SRCS = $(SRC_DIR)/main.cu \
       $(SRC_DIR)/radix_cuda.cu \
       $(SRC_DIR)/radix_cpu.cpp \
       $(SRC_DIR)/utils.cpp

.PHONY: all clean

all:
	$(NVCC) $(NVCCFLAGS) $(SRCS) -o $(EXEC)

clean:
	del /Q $(EXEC) 2>nul || exit 0
