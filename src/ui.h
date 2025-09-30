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

void ui_scissor(float x, float y, float w, float h);

void ui_window(size_t x, size_t y, size_t width, size_t height, uint32_t background);


#endif