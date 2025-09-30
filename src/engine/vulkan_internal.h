#ifndef TRIEX_VULKAN_INTERNAL
#define TRIEX_VULKAN_INTERNAL

int getNeededQueueFamilyIndex(VkQueueFlags flags);
bool findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t* index);

#endif