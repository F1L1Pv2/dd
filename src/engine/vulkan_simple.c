#include "vulkan_simple.h"

#include "engine/vulkan_init.h"
#include "engine/vulkan_initSurface.h"
#include "engine/vulkan_initDevice.h"
#include "engine/vulkan_initSwapchain.h"
#include "engine/vulkan_initCommandPool.h"
#include "engine/vulkan_initDescriptorPool.h"

bool vulkan_init_with_window(const char* title, size_t width, size_t height){
    platform_create_window(title, width, height);
    if(!initVulkan()) return false;
    if(!initSurface()) return false;
    if(!initDevice()) return false;
    if(!initSwapchain()) return false;
    if(!initCommandPool()) return false;
    if(!initDescriptorPool()) return false;
    return true;
}

bool vulkan_init_headless(){
    if(!initVulkan()) return false;
    if(!initDevice()) return false;
    if(!initCommandPool()) return false;
    if(!initDescriptorPool()) return false;
    return true;
}