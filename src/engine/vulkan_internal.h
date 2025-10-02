#ifndef TRIEX_VULKAN_INTERNAL
#define TRIEX_VULKAN_INTERNAL

extern VkPhysicalDeviceMemoryProperties physicalMemoryProperties;
extern VkPhysicalDevice physicalDevice;
extern VkPhysicalDeviceLimits physicalDeviceLimits;
extern VkInstance instance;

typedef struct{
    VkQueueFamilyProperties* items;
    uint32_t count;
    size_t capacity;
} MultipleVkQueueFamilyProperties;
extern MultipleVkQueueFamilyProperties multipleQueueFamilyProperties;

int getNeededQueueFamilyIndex(VkQueueFlags flags);
bool findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties, uint32_t* index);

#endif