#ifndef TRIEX_VULKAN_GLOBALS
#define TRIEX_VULKAN_GLOBALS

#include "vulkan/vulkan.h"

extern VkDevice device;
extern VkSurfaceKHR surface;
extern VkSwapchainKHR swapchain;
extern VkFormat swapchainImageFormat;
extern VkExtent2D swapchainExtent;
extern VkQueue graphicsQueue;
extern VkQueue presentQueue;

typedef struct {
    VkImage* items;
    uint32_t count;
    size_t capacity;
} VkImages;

typedef struct {
    VkImageView* items;
    uint32_t count;
    size_t capacity;
} VkImageViews;

extern VkImages swapchainImages;
extern VkImageViews swapchainImageViews;

extern VkCommandPool commandPool;
extern VkDescriptorPool descriptorPool;

#endif