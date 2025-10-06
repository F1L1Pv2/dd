#include <stdio.h>

#include "engine/vulkan_simple.h"
#include "thirdparty/stb_image.h"

#include "dd.h"
#include <math.h>
#include <string.h>

int main(){
    vulkan_init_with_window("TRIEX NEW!", 640, 480);

    VkCommandBuffer cmd;
    if(vkAllocateCommandBuffers(device,&(VkCommandBufferAllocateInfo){
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = NULL,
        .commandPool = commandPool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    },&cmd) != VK_SUCCESS) return 1;

    VkShaderModule vertexShader;
    const char* vertexShaderSrc = 
        "#version 450\n"
        "layout(location = 0) in vec2 inPosition;\n"
        "vec3 colors[3] = vec3[](vec3(1.0, 0.0, 0.0),vec3(0.0, 1.0, 0.0),vec3(0.0, 0.0, 1.0));"
        "layout(location = 0) out vec3 color;\n"
        "void main() {\n"
        "   color = colors[gl_VertexIndex]; \n"
        "   gl_Position = vec4(inPosition,0.0f,1.0f); \n"
        "}";

    if(!vkCompileShader(device, vertexShaderSrc, shaderc_vertex_shader,&vertexShader)) return 1;

    VkShaderModule fragmentShader;
    const char* fragmentShaderSrc = 
        "#version 450\n"
        "layout(location = 0) out vec4 outColor;\n"
        "layout(location = 0) in vec3 inColor;\n"
        "void main() {\n"
        "   outColor = vec4(inColor,1.0f);"
        "\n}";
    if(!vkCompileShader(device, fragmentShaderSrc, shaderc_fragment_shader,&fragmentShader)) return 1;

    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    if(!vkCreateGraphicPipeline(
        vertexShader, fragmentShader,
        &pipeline,
        &pipelineLayout,
        swapchainImageFormat,
        .vertexSize = sizeof(float)*2,
        .vertexInputAttributeDescriptionsCount = 1,
        .vertexInputAttributeDescriptions = &(VkVertexInputAttributeDescription){
            .location = 0,
            .binding = 0,
            .format = VK_FORMAT_R32G32_SFLOAT,
            .offset = 0,
        },
    )) return 1;

    VkFence renderingFence;
    if(vkCreateFence(device,
        &(VkFenceCreateInfo){
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT 
        },
        NULL,
        &renderingFence
    ) != VK_SUCCESS) return 1;

    VkSemaphore swapchainHasImageSemaphore;
    if(vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}, NULL, &swapchainHasImageSemaphore) != VK_SUCCESS) return 1;
    VkSemaphore readyToSwapYourChainSemaphore;
    if(vkCreateSemaphore(device, &(VkSemaphoreCreateInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO}, NULL, &readyToSwapYourChainSemaphore) != VK_SUCCESS) return 1;

    float vertices[] = {
        0.0, -0.5,
        -0.5, 0.5,
        0.5, 0.5
    };
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, sizeof(vertices), &vertexBuffer, &vertexBufferMemory)) return 1;
    {
        void* mapped;
        if(vkMapMemory(device, vertexBufferMemory, 0, sizeof(vertices), 0, &mapped) != VK_SUCCESS) return 1;
        memcpy(mapped, vertices, sizeof(vertices));
        vkUnmapMemory(device, vertexBufferMemory);
    }

    if(!dd_init(device, swapchainImageFormat, descriptorPool)) return 1;

    uint32_t image_id;
    uint32_t image2_id;
    uint32_t image3_id;

    {
        int w,h;
        void* data = stbi_load("assets/test.png", &w, &h, NULL, 4);
        if(data == NULL) {
            fprintf(stderr, "Couldn't load image!\n");
            return 1;
        }
        
        image_id = dd_create_texture(w,h);
        if(image_id == -1) {
            fprintf(stderr, "Couldn't create texture!\n");
            return 1;
        }

        dd_update_texture(image_id, data);
        
        stbi_image_free(data);
    }

    {
        int w,h;
        void* data = stbi_load("assets/test2.png", &w, &h, NULL, 4);
        if(data == NULL) {
            fprintf(stderr, "Couldn't load image!\n");
            return 1;
        }
        
        image2_id = dd_create_texture(w,h);
        if(image_id == -1) {
            fprintf(stderr, "Couldn't create texture!\n");
            return 1;
        }

        dd_update_texture(image2_id, data);
        
        stbi_image_free(data);
    }

    image3_id = dd_create_texture(16,16);
    void* image3_mapped = dd_map_texture(image3_id);
    size_t image3_stride = dd_get_texture_stride(image3_id);
    memset(image3_mapped, 0xFFFFFFFF, image3_stride*16);

    uint32_t imageIndex;

    uint64_t oldTime = platform_get_time_nanos();
    double time = 0.0;
    double target_fps = 60.0;
    double target_frame_time = 1.0 / target_fps;
    bool using = false;
    double counter = 0.0;
    while(platform_still_running()){
        platform_window_handle_events();
        if(platform_window_minimized){
            platform_sleep(1);
            continue;
        }

        uint64_t now = platform_get_time_nanos();
        double dt = (double)(now - oldTime) * 1e-9;
        oldTime = now;

        if (dt > 0.1) dt = 0.1;

        time += dt;
        counter += dt;

        dd_begin();

        counter += dt;

        if(counter > 1.0) {
            using = !using;
            counter = 0;
        }

        dd_scissor(0,0,swapchainExtent.width*(cos(time)/2+0.5), swapchainExtent.height*(sin(time)/2+0.5));
        size_t w = swapchainExtent.height;
        dd_image(using ? image2_id : 69,swapchainExtent.width/2 - w/2,swapchainExtent.height/2 - w/2,w,w, 0,0, 1, 1, 0xFFFFFFFF);

        dd_scissor(0,0,0,0);
        w = swapchainExtent.height/2;
        dd_image(image_id,swapchainExtent.width/2 - sin(time) * ((sin(time)/2+0.5)*200) - w/2,swapchainExtent.height/2 - cos(time) * ((sin(time)/2+0.5)*200) - w/2,w,w, 0,0, 1, 1, 0xFFFFFFFF);

        dd_scissor(0,swapchainExtent.height/2-swapchainExtent.height/4,swapchainExtent.width, swapchainExtent.height/2);

        dd_rect(swapchainExtent.width/2 - sin(time) * 200 - 100,swapchainExtent.height/2 - 100,200,200, 0xFFFF0000);

        dd_rect(swapchainExtent.width/2 - 100,swapchainExtent.height/2 - sin(time) * 200 - 100,200,200, 0xFF00FF00);

        dd_scissor(0,0,swapchainExtent.width/2, swapchainExtent.height);

        dd_rect(swapchainExtent.width/2 + cos(time) * 200 - 100,swapchainExtent.height/2 + sin(time) * 200 - 100,200,200, 0xFF0000FF);

        dd_scissor(swapchainExtent.width/2,0,swapchainExtent.width/2, swapchainExtent.height);

        dd_rect(swapchainExtent.width/2 + sin(time) * 200 - 100,swapchainExtent.height/2 + cos(time) * 200 - 100,200,200, 0xFFFFFF00);

        dd_scissor(0,0,0,0);

        for(size_t y = 0; y < 16; y++){
            for(size_t x = 0; x < 16; x++){
                uint32_t* pixel = (uint32_t*)((uint8_t*)image3_mapped + y * image3_stride + x * sizeof(uint32_t));
                *pixel = 0xFF000000 | rand(); // RGBA pixel
            }
        }

        dd_image(image3_id, (sin(time)/2+0.5)*(swapchainExtent.width-128),swapchainExtent.height/2,128,128,0,0,1,1,0xFFFFFFFF);

        {
            const char* text = "Hello Baller!\nF1L1P Here!";
            float size = swapchainExtent.height/20 * (sin(time)/2+0.5)*2; 
            dd_text(text, swapchainExtent.width/2 - dd_text_measure(text, size)/2, swapchainExtent.height/2, size, 0xFFFFFFFF);
        }

        {
            char text[256];
            snprintf(text, sizeof(text), "time: %f\ndt: %f", time, dt);
            float size = swapchainExtent.height/15; 
            dd_text(text, 0, 0, size, 0xFFFFFFFF);
        }

        dd_end();

        vkWaitForFences(device, 1, &renderingFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &renderingFence);
        vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainHasImageSemaphore, NULL, &imageIndex);

        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO});
        
        vkCmdTransitionImage(cmd, swapchainImages.items[imageIndex], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT);

        vkCmdBeginRenderingEX(cmd,
            .colorAttachment = swapchainImageViews.items[imageIndex],
            .clearColor = COL_HEX(0xFF181818),
            .renderArea = (
                (VkExtent2D){.width = swapchainExtent.width, .height = swapchainExtent.height}
            )
        );

        vkCmdSetViewport(cmd, 0, 1, &(VkViewport){
            .width = swapchainExtent.width,
            .height = swapchainExtent.height
        });
            
        vkCmdSetScissor(cmd, 0, 1, &(VkRect2D){
            .extent.width = swapchainExtent.width,
            .extent.height = swapchainExtent.height,
        });

        vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertexBuffer, &(VkDeviceSize){0});
        vkCmdDraw(cmd, 3, 1, 0, 0);

        vkCmdEndRendering(cmd);

        dd_draw(cmd, swapchainExtent.width, swapchainExtent.height, swapchainImageViews.items[imageIndex]);

        vkCmdTransitionImage(cmd, swapchainImages.items[imageIndex], VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        vkEndCommandBuffer(cmd);

        vkQueueSubmit(graphicsQueue, 1, &(VkSubmitInfo){
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &cmd,
            
            .waitSemaphoreCount = 1,
            .pWaitDstStageMask = &(VkPipelineStageFlags){VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT},
            .pWaitSemaphores = &swapchainHasImageSemaphore,

            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &readyToSwapYourChainSemaphore,
        }, renderingFence);

        vkQueuePresentKHR(presentQueue, &(VkPresentInfoKHR){
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &readyToSwapYourChainSemaphore,

            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &imageIndex
        });

        uint64_t frameEnd = platform_get_time_nanos();
        double frameTime = (double)(frameEnd - now) * 1e-9;
        if (frameTime < target_frame_time) {
            platform_sleep((target_frame_time - frameTime)*1000);
        }
    }

    return 0;
}