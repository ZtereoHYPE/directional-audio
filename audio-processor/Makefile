# Compiler and flags
SHADER_COMPILER := glslc

# Directories
SHADER_DIR := src/shaders
SUBDIR_SHADER_FILES := $(shell find $(SHADER_DIR) -mindepth 2 -name "*.comp")
SHADER_BUILD_DIR := target/shaders

# Shader files
COMPUTE_SHADERS := $(wildcard $(SHADER_DIR)/*.comp)
COMPILED_COMPUTE_SHADERS := $(COMPUTE_SHADERS:$(SHADER_DIR)/%.comp=$(SHADER_BUILD_DIR)/%.comp.spv)

# Rules
shaders: $(COMPILED_COMPUTE_SHADERS)

$(SHADER_BUILD_DIR)/%.comp.spv: $(SHADER_DIR)/%.comp $(SUBDIR_SHADER_FILES)
	@mkdir -p $(SHADER_BUILD_DIR)
	$(SHADER_COMPILER) $< -o $@

clean:
	rm -rf $(SHADER_BUILD_DIR)

.PHONY: all clean
