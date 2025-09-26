#ifndef TRIEX_VULKAN_HELPERS
#define TRIEX_VULKAN_HELPERS

#include <stdbool.h>

#include "vulkan/vulkan.h"
int getNeededQueueFamilyIndex(VkQueueFlags flags);
bool findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t* index);
VkCommandBuffer beginSingleTimeCommands();
void endSingleTimeCommands(VkCommandBuffer commandBuffer);
void vkCmdTransitionImage(VkCommandBuffer commandBuffer, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlagBits oldStage, VkPipelineStageFlagBits newStage);

typedef struct {
    float r;
    float g;
    float b;
    float a;
} Color;

typedef struct{
    VkImageView colorAttachment;
    Color clearColor;
    bool clearBackground;
    VkImageView depthAttachment;
    VkExtent2D renderArea;
} BeginRenderingEX;

#define COL_BLACK ((Color){0.0,0.0,0.0,1.0})
#define COL_EMPTY ((Color){0.0,0.0,0.0,0.0})
#define COL_HEX(c) ((Color){ \
    .r = ((float)(((c) >> 16) & 0xFF) / 255.0f), \
    .g = ((float)(((c) >>  8) & 0xFF) / 255.0f), \
    .b = ((float)(((c) >>  0) & 0xFF) / 255.0f), \
    .a = ((float)(((c) >> 24) & 0xFF) / 255.0f)  \
})

void vkCmdBeginRenderingEX_opt(VkCommandBuffer commandBuffer, BeginRenderingEX args);

#define vkCmdBeginRenderingEX(cmd, ...) vkCmdBeginRenderingEX_opt(cmd, (BeginRenderingEX){.clearBackground = true,__VA_ARGS__})

#endif