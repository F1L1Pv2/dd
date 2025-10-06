#ifndef DD_DD
#define DD_DD

#include "vulkan/vulkan.h"
#include <stdbool.h>

bool dd_init(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool);
void dd_begin();

void dd_end();

void dd_draw(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment);

void dd_rect(float x, float y, float w, float h, uint32_t color);

void dd_text(const char* text, float x, float y, float size, uint32_t color);

float dd_text_measure(const char* text, float size);

void dd_image(uint32_t texture_id, float x, float y, float w, float h, float uv_x, float uv_y, float uv_w, float uv_h, uint32_t albedo);

uint32_t dd_create_texture(size_t width, size_t height); // returns texture id (-1 on failure)
bool dd_update_texture(uint32_t texture_id, void* data); // data is in uint32_t rgba
bool dd_destroy_texture(uint32_t texture_id);

void dd_scissor(float x, float y, float w, float h);

#endif