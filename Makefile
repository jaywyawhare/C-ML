# Top-level Makefile: delegates to CMake in build/
# Usage: make [clean]

BUILD_DIR := build
JOBS := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

.PHONY: all clean

all:
	@mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		$(MAKE) -j$(JOBS)

clean:
	@rm -rf $(BUILD_DIR)
