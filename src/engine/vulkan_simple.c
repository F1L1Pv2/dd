#include "vulkan_simple.h"

#include "engine/vulkan_initialize.h"
#include "engine/vulkan_createSurface.h"
#include "engine/vulkan_getDevice.h"
#include "engine/vulkan_initSwapchain.h"
#include "engine/vulkan_initCommandPool.h"
#include "engine/vulkan_initCommandBuffer.h"
#include "engine/vulkan_initDescriptorPool.h"
#include "engine/vulkan_initSamplers.h"

bool vulkan_init_with_window(const char* title, size_t width, size_t height){
    platform_create_window(title, width, height);
    if(!initialize_vulkan()) return false;
    if(!createSurface()) return false;
    if(!getDevice()) return false;
    if(!initSwapchain()) return false;
    if(!initCommandPool()) return false;
    if(!initCommandBuffer()) return false;
    if(!initDescriptorPool()) return false;
    if(!initSamplers()) return false;
    return true;
}

bool vulkan_init_headless(){
    if(!initialize_vulkan()) return false;
    if(!getDevice()) return false;
    if(!initCommandPool()) return false;
    if(!initCommandBuffer()) return false;
    if(!initDescriptorPool()) return false;
    if(!initSamplers()) return false;
    return true;
}