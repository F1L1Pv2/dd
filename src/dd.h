#ifndef DD_DD
#define DD_DD

#include "vulkan/vulkan.h"
#include <stdbool.h>

bool dd_init(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool);
void dd_begin();

void dd_end();

void dd_draw(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment);

void dd_rect(float x, float y, float w, float h, uint32_t color);

void dd_rotated_rect(float x, float y, float w, float h, float angle, uint32_t color); // origin is at top left corner

void dd_circle(float x, float y, float radius, uint32_t color);

void dd_line(float x1, float y1, float x2, float y2, float thickness, uint32_t color);

void dd_bezier_cubic(float x1, float y1, float x2, float y2, float cx1, float cy1, float cx2, float cy2, float thickness, int segments, uint32_t color);

void dd_bezier_quadratic(float x1, float y1, float x2, float y2, float cx, float cy, float thickness, int segments, uint32_t color);

void dd_text(const char* text, float x, float y, float size, uint32_t color);

float dd_text_measure(const char* text, float size);

void dd_image(uint32_t texture_id, float x, float y, float w, float h, float uv_x, float uv_y, float uv_w, float uv_h, uint32_t albedo);

uint32_t dd_create_texture(size_t width, size_t height); // returns texture id (-1 on failure)
bool dd_update_texture(uint32_t texture_id, void* data); // data is in uint32_t rgba
bool dd_destroy_texture(uint32_t texture_id);
void* dd_map_texture(uint32_t texture_id);
void dd_unmap_texture(uint32_t texture_id);
size_t dd_get_texture_stride(uint32_t texture_id);

void dd_scissor(float x, float y, float w, float h);

#endif