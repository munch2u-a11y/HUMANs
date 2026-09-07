PROJECT_ROOT := $(abspath ../..)
MODEL ?= $(PROJECT_ROOT)/models/Qwen3-0.6B-Q8_0.gguf
LLAMA_CPP_SOURCE ?= $(PROJECT_ROOT)/third_party/llama.cpp
OLLAMA_LIB_DIR ?=

.PHONY: build live opaque nursery embodied-nursery reverse-nursery gestate-fast transformer-hatch open-weight cortex-rocm-setup cortex-rocm-probe cortex-nursery communication-nursery test

build:
	$(MAKE) -C native LLAMA_CPP_SOURCE="$(LLAMA_CPP_SOURCE)" OLLAMA_LIB_DIR="$(OLLAMA_LIB_DIR)"

live: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/live_tester.py --model "$(MODEL)" --show-trace

opaque: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/opaque_skeleton.py --model "$(MODEL)"

nursery: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/nursery.py --model "$(MODEL)"

embodied-nursery:
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/embodied_nursery.py

reverse-nursery: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/reverse_nursery.py --model "$(MODEL)"

gestate-fast: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/accelerated_gestation.py --model "$(MODEL)"

transformer-hatch: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/transformer_hatch.py --model "$(MODEL)"

open-weight: build
	cd ../.. && PYTHONPATH=src python3 experiments/graph_native_live/unified_open_weight_agent.py --model "$(MODEL)"

cortex-rocm-setup:
	cd ../.. && experiments/graph_native_live/setup_rocm_cortex.sh

cortex-rocm-probe:
	cd ../.. && PYTHONPATH=src .venv-rocm/bin/python experiments/graph_native_live/rocm_cortex_probe.py

cortex-nursery:
	cd ../.. && PYTHONPATH=src .venv-rocm/bin/python experiments/graph_native_live/developmental_cortex_nursery.py

communication-nursery:
	cd ../.. && PYTHONPATH=src .venv-rocm/bin/python experiments/graph_native_live/developmental_communication_nursery.py --tiny

test:
	cd ../.. && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m pytest -q tests/test_graph_native_live.py tests/test_opaque_graph_native.py tests/test_nursery.py tests/test_embodied_nursery.py tests/test_reverse_nursery.py tests/test_accelerated_gestation.py tests/test_open_weight_runtime.py tests/test_desire_nursery.py tests/test_self_pulse_kernel.py tests/test_trunk_breadth_growth.py tests/test_developmental_curriculum.py tests/test_developmental_cortex.py tests/test_developmental_cortex_nursery.py tests/test_developmental_communication_nursery.py
