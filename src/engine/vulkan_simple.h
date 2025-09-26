#ifndef TRIEX_VULKAN_SIMPLE
#define TRIEX_VULKAN_SIMPLE

#include "vulkan/vulkan.h"
#include "engine/platform.h"
#include "engine/vulkan_helpers.h"
#include "engine/vulkan_compileShader.h"
#include "engine/vulkan_buffer.h"
#include "engine/vulkan_createGraphicPipelines.h"
#include "engine/vulkan_globals.h"
#include "engine/input.h"

#include <stdbool.h>
bool vulkan_init_with_window(const char* title, size_t width, size_t height);
bool vulkan_init_headless();

#endif