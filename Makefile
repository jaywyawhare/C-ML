# Top-level Makefile: delegates to CMake in build/
# Usage: make | make asan | make clean

BUILD_DIR := build
BUILD_ASAN_DIR := build-asan
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all asan clean clean-asan

all:
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		$(MAKE) -j$(JOBS)

asan:
	@mkdir -p $(BUILD_ASAN_DIR) && cd $(BUILD_ASAN_DIR) && \
		cmake .. -DCMAKE_BUILD_TYPE=Debug -DENABLE_SANITIZERS=ON && \
		$(MAKE) -j$(JOBS)

clean:
	@rm -rf $(BUILD_DIR)

clean-asan:
	@rm -rf $(BUILD_ASAN_DIR)
