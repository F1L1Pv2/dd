#ifndef UI_UI
#define UI_UI

#include "vulkan/vulkan.h"
#include <stdbool.h>

bool ui_init(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool);
void ui_begin(
    size_t mouseX, size_t mouseY, 
    bool mouse_left_down,
    bool mouse_left_justPressed,
    bool mouse_left_justReleased,

    bool mouse_middle_down,
    bool mouse_middle_justPressed,
    bool mouse_middle_justReleased,

    bool mouse_right_down,
    bool mouse_right_justPressed,
    bool mouse_right_justReleased,
    double mouse_scroll,

    char* lastTextKey // used for typing in text boxes
);

void ui_end();

void ui_draw(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment);

void ui_rect(float x, float y, float w, float h, uint32_t color);

void ui_text(const char* text, float x, float y, float size, uint32_t color);

float ui_text_measure(const char* text, float size);

void ui_image(uint32_t texture_id, float x, float y, float w, float h, float uv_x, float uv_y, float uv_w, float uv_h, uint32_t albedo);

uint32_t ui_create_texture(size_t width, size_t height); // returns texture id (-1 on failure)
bool ui_update_texture(uint32_t texture_id, void* data); // data is in uint32_t rgba
bool ui_destroy_texture(uint32_t texture_id);

void ui_scissor(float x, float y, float w, float h);

#endif