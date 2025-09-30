#ifndef TRIEX_VULKAN_IMAGES
#define TRIEX_VULKAN_IMAGES

#include <stdbool.h>
#include <vulkan/vulkan.h>

bool vkCreateImageViewEX(VkDevice device, VkImage image, VkFormat format, VkImageAspectFlags aspectFlags, VkImageView* out);
bool vkCreateImageEX(VkDevice device, size_t width, size_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags memoryProperties, VkImage* image, VkDeviceMemory* imageMemory);
size_t vkGetImageStride(VkDevice device, VkImage image);

#endif