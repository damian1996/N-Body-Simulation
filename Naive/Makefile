CUDA_INSTALL_PATH ?= /usr/local/cuda-9.2/bin
CXX := /usr/bin/g++
LDFLAGS=-lglfw -lGL
NVCC  := $(CUDA_INSTALL_PATH)/nvcc
BUILDDIR := build
OUTDIR := out
SRCDIR := src
SOURCES := $(wildcard src/*.cpp)
CUSOURCES := $(wildcard src/*.cu)
OBJS := $(subst .cpp,.cpp.o,$(SOURCES))
OBJS := $(subst src,build,$(OBJS))
CUOBJS := $(subst .cu,.o,$(CUSOURCES))
CUOBJS := $(subst src,build,$(CUOBJS))
TARGET := $(OUTDIR)/main

# Options
NVCCOPTIONS = -m64 -std=c++14 -g
CXXOPTIONS = -m64 -std=c++14 -g -Wall

# Common flags
NVCCFLAGS += $(NVCCOPTIONS)
CXXFLAGS += $(CXXOPTIONS)

all: $(TARGET)

build/%.cpp.o: src/%.cpp src/%.h
	@mkdir -p $(BUILDDIR)
	$(CXX) $< $(CXXFLAGS) -c -o $@

$(TARGET): $(OBJS) $(CUOBJS)
	@mkdir -p $(OUTDIR)
	$(NVCC) $^ $(LDFLAGS) -o $@

build/%.o: src/%.cu src/%.cuh
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	@rm -rf $(OUTDIR) $(BUILDDIR)

printVar:
	$(info value of var is $(CUOBJS))
