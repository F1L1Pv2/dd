#ifdef _WIN32
    #include "platform_windows.h"
#else
    #include "platform_linux.h"
#endif

#include "vulkan_globals.h"
#include "vulkan_initSwapchain.h"
#include "stdio.h"

bool platform_window_minimized = false;
bool platform_resize_window_callback(bool minimized){
    platform_window_minimized = minimized;

    if(platform_window_minimized) return true;

    if(!swapchain) return true;
    VkResult result = vkDeviceWaitIdle(device);
    if(result != VK_SUCCESS){
        fprintf(stderr, "ERROR: Couldn't wait for fences\n");
        return false;
    }

    for(int i = 0; i < swapchainImageViews.count; i++){
        vkDestroyImageView(device,swapchainImageViews.items[i],NULL);
    }
    swapchainImages.count = 0;
    swapchainImageViews.count = 0;
    vkDestroySwapchainKHR(device, swapchain, NULL);

    if(!initSwapchain()) return false;

    return true;
}