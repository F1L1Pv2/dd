#ifndef TRIEX_VULKAN_BUFFER
#define TRIEX_VULKAN_BUFFER

#include <stdbool.h>
#include <vulkan/vulkan.h>

bool vkCreateBufferEX(VkDevice device, VkBufferUsageFlagBits usage, VkMemoryPropertyFlagBits memoryProperties, VkDeviceSize size, VkBuffer* outBuff, VkDeviceMemory* outMemory);

#endif