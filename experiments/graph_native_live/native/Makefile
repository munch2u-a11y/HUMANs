CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
PROJECT_ROOT := $(abspath ../../..)
LLAMA_CPP_SOURCE ?= $(PROJECT_ROOT)/third_party/llama.cpp
OLLAMA_LIB_DIR ?=

COMMA := ,
INCLUDES = -I"$(LLAMA_CPP_SOURCE)/include" -I"$(LLAMA_CPP_SOURCE)/ggml/include"
LIB_SEARCH = $(if $(strip $(OLLAMA_LIB_DIR)),-L"$(OLLAMA_LIB_DIR)" -Wl$(COMMA)-rpath$(COMMA)"$(OLLAMA_LIB_DIR)",)
LIBS = $(LIB_SEARCH) \
	-lllama -lggml -lggml-base -ldl -pthread

.PHONY: all clean

all: graph_soft_generator lexeme_codec

graph_soft_generator: graph_soft_generator.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) "$<" $(LIBS) -o "$@"

lexeme_codec: lexeme_codec.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) "$<" $(LIBS) -o "$@"

clean:
	rm -f graph_soft_generator lexeme_codec
