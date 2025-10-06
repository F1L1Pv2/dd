#include "dd.h"

#include "vulkan/vulkan.h"
#include "engine/vulkan_createGraphicPipelines.h"
#include "engine/vulkan_compileShader.h"
#include "engine/vulkan_helpers.h"
#include "engine/vulkan_buffer.h"
#include "engine/vulkan_images.h"
#include <stdbool.h>
#include <string.h>
#include <stdlib.h>
#include "thirdparty/stb_image.h"

#ifndef DEBUG
#define assert(...)
#else
#include <assert.h>
#endif

#define DA_REALLOC(optr, osize, new_size) realloc(optr, new_size)
#define da_reserve(da, extra) \
   do {\
      if((da)->count + extra >= (da)->capacity) {\
          void* _da_old_ptr;\
          size_t _da_old_capacity = (da)->capacity;\
          (void)_da_old_capacity;\
          (void)_da_old_ptr;\
          (da)->capacity = (da)->capacity*2+extra;\
          _da_old_ptr = (da)->items;\
          (da)->items = DA_REALLOC(_da_old_ptr, _da_old_capacity*sizeof(*(da)->items), (da)->capacity*sizeof(*(da)->items));\
          assert((da)->items && "Ran out of memory");\
      }\
   } while(0)
#define da_push(da, value) \
   do {\
        da_reserve(da, 1);\
        (da)->items[(da)->count++]=value;\
   } while(0)

#define da_arena_reserve(da, extra) \
   do {\
      if((da)->size + extra >= (da)->capacity) {\
          void* _da_old_ptr;\
          size_t _da_old_capacity = (da)->capacity;\
          (void)_da_old_capacity;\
          (void)_da_old_ptr;\
          (da)->capacity = (da)->capacity*2+extra;\
          _da_old_ptr = (da)->buff;\
          (da)->buff = DA_REALLOC(_da_old_ptr, _da_old_capacity, (da)->capacity);\
          assert((da)->buff && "Ran out of memory");\
      }\
   } while(0)

#define da_arena_push(da, value) \
   do {\
        da_arena_reserve(da, sizeof(value));\
        memcpy(((uint8_t*)(da)->buff) + (da)->size, &value, sizeof(value));\
        (da)->size += sizeof(value);\
   } while(0)

typedef struct{
    float x;
    float y;
} vec2;

typedef struct{
    float x;
    float y;
    float z;
    float w;
} vec4;

typedef struct {
    float v[16];
} mat4;

static mat4 ortho2D(float width, float height){
    float left = -width/2;
    float right = width/2;
    float top = height/2;
    float bottom = -height/2;

    return (mat4){
    2 / (right - left),0                 , 0, -(right + left) / (right - left),
          0           ,2 / (top - bottom), 0, -(top + bottom) / (top - bottom),
          0           ,     0            ,-1,                 0,
          0           ,     0            , 0,                 1,
    };
}

static mat4 mat4mul(mat4 *a, mat4 *b) {
    mat4 result;

    // Column-major multiplication: result[i][j] = sum_k a[k][j] * b[i][k]
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k) {
                // a is row × k, b is k × col
                float a_elem = a->v[k * 4 + row]; // a[row][k]
                float b_elem = b->v[col * 4 + k]; // b[k][col]
                sum += a_elem * b_elem;
            }
            result.v[col * 4 + row] = sum; // result[row][col]
        }
    }

    return result;
}

// --------------------------- actual ui stuff -------------------------------------
typedef struct{
    vec2 offset;
    vec2 size;
} ScissorDrawCommand;

typedef struct{
    vec2 position;
    vec2 scale;
    vec4 albedo;
} RectDrawCommand;

typedef struct {
    vec2 position;
    vec2 scale;
    vec2 uv_offset;
    vec2 uv_size;
    vec4 albedo;
} TextDrawCommand;

typedef struct {
    vec2 position;   // 8 bytes
    vec2 scale;      // 8 bytes
    vec2 uv_offset;  // 8 bytes
    vec2 uv_size;    // 8 bytes
    vec4 albedo;     // 16 bytes
    uint32_t texture_id;  // 4 bytes

    // Explicit padding to bring stride up to 64 bytes
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
} ImageDrawCommand;

typedef enum {
    DD_DRAW_CMD_NONE = 0,
    DD_DRAW_CMD_RECT,
    DD_DRAW_CMD_SCISSOR,
    DD_DRAW_CMD_TEXT,
    DD_DRAW_CMD_IMAGE,
    DD_DRAW_CMDS_COUNT,
} UiDrawCmdType;

typedef struct{
    UiDrawCmdType type;
    union{
        RectDrawCommand rect;
        ScissorDrawCommand scissor;
        TextDrawCommand text;
        ImageDrawCommand image;
    } as;
} DrawCommand;

typedef struct {
    void* buff;
    size_t size;
    size_t capacity;
} DrawCommandArena;

typedef struct{
    UiDrawCmdType* items;
    size_t count;
    size_t capacity;
} DrawCommands;

typedef struct{
    mat4 projView;
} PushConstants;

static bool inited = false;
static VkSampler dd_samplerNearest;
static DrawCommands drawCommands = {0};
static DrawCommandArena drawCommandsArena = {0};
static PushConstants pushConstants;

static bool dd_create_buffer_descriptor_set_and_bind_buffer(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet* descriptorSetOut, VkBuffer buffer, VkDeviceSize size){
    VkResult result;
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {0};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;
    
    result = vkAllocateDescriptorSets(device,&descriptorSetAllocateInfo, descriptorSetOut);
    if(result != VK_SUCCESS) return false;

    VkDescriptorBufferInfo descriptorBufferInfo = {
        .buffer = buffer,
        .offset = 0,
        .range = size,
    };

    VkWriteDescriptorSet writeDescriptorSet = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .dstSet = *descriptorSetOut,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .pBufferInfo = &descriptorBufferInfo
    };
        
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);

    return true;
}

static bool dd_create_image_descriptor_set_and_bind_image(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet* descriptorSetOut, VkImageView imageView, VkSampler sampler){
    VkResult result;
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {0};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &descriptorSetLayout;
    
    result = vkAllocateDescriptorSets(device,&descriptorSetAllocateInfo, descriptorSetOut);
    if(result != VK_SUCCESS) return false;

    VkDescriptorImageInfo descriptorImageInfo = {
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        .imageView = imageView,
        .sampler = sampler,
    };

    VkWriteDescriptorSet writeDescriptorSet = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .dstSet = *descriptorSetOut,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .pImageInfo = &descriptorImageInfo,
    };
        
    vkUpdateDescriptorSets(device, 1, &writeDescriptorSet, 0, NULL);

    return true;
}

static void dd_transitionMyImage_inner(VkCommandBuffer tempCmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlagBits oldStage, VkPipelineStageFlagBits newStage){
    VkImageMemoryBarrier barrier = {0};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.pNext = NULL;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(
        tempCmd,
        oldStage,
        newStage,
        0,
        0, NULL,
        0, NULL,
        1, &barrier
    );
}

static void dd_transitionMyImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlagBits oldStage, VkPipelineStageFlagBits newStage){
    VkCommandBuffer tempCmd = vkCmdBeginSingleTime();
    dd_transitionMyImage_inner(tempCmd, image, oldLayout, newLayout, oldStage, newStage);
    vkCmdEndSingleTime(tempCmd);
}

static bool dd_createMyImage(VkDevice device, VkImage* image, size_t width, size_t height, VkDeviceMemory* imageMemory, VkImageView* imageView, size_t* imageStride, void** imageMapped, VkImageUsageFlagBits imageUsage, VkMemoryPropertyFlagBits memoryProperty){
    if(!vkCreateImageEX(device, width, height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_LINEAR,
            imageUsage,
            memoryProperty, image,imageMemory)){
        printf("Couldn't create image\n");
        return false;
    }

    if(!vkCreateImageViewEX(device,*image,VK_FORMAT_R8G8B8A8_UNORM, 
                VK_IMAGE_ASPECT_COLOR_BIT, imageView)){
        printf("Couldn't create image view\n");
        return false;
    }

    *imageStride = vkGetImageStride(device, *image);
    if(imageMapped) vkMapMemory(device,*imageMemory, 0, (*imageStride)*height, 0, imageMapped);

    return true;
}

#define MAX_RECT_COUNT 128
static VkPipeline rectPipeline;
static VkPipelineLayout rectPipelineLayout;
static VkDescriptorSetLayout rectDescriptorSetLayout = {0};

static bool dd_init_rects(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
    //TODO use precompiled shaders
    VkShaderModule vertexShader;
    const char* vertexShaderSrc =
            "#version 450\n"
            "#extension GL_EXT_scalar_block_layout : require\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "struct RectDrawCommand {\n"
            "    vec2 position;\n"
            "    vec2 scale;\n"
            "    vec4 albedo;\n"
            "};\n"
            "layout(set = 0, binding = 0, scalar) readonly buffer RectDrawBuffer {\n"
            "    RectDrawCommand commands[];\n"
            "};\n"
            "layout(push_constant) uniform Constants {\n"
            "    mat4 projView;\n"
            "} pcs;\n"
            "layout(location = 1) out flat uint InstanceIndex;\n"
            "void main() {\n"
            "    uint b = 1 << (gl_VertexIndex % 6);\n"
            "    vec2 baseCoord = vec2((0x1C & b) != 0, (0xE & b) != 0);\n"
            "    vec2 pos = commands[gl_InstanceIndex].position;\n"
            "    vec2 scale = commands[gl_InstanceIndex].scale;\n"
            "    mat4 model = mat4(\n"
            "        vec4(scale.x, 0,       0, 0),\n"
            "        vec4(0,       scale.y, 0, 0),\n"
            "        vec4(0,       0,       1, 0),\n"
            "        vec4(pos.x,   pos.y,   0, 1)\n"
            "    );\n"
            "    gl_Position = pcs.projView * model * vec4(baseCoord, 0.0, 1.0);\n"
            "    InstanceIndex = gl_InstanceIndex;\n"
            "}\n";

        if(!vkCompileShader(device, vertexShaderSrc, shaderc_vertex_shader,&vertexShader)) return false;

    VkShaderModule fragmentShader;
    const char* fragmentShaderSrc =
            "#version 450\n"
            "#extension GL_EXT_scalar_block_layout : require\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "struct RectDrawCommand {\n"
            "    vec2 position;\n"
            "    vec2 scale;\n"
            "    vec4 albedo;\n"
            "};\n"
            "layout(set = 0, binding = 0, scalar) readonly buffer RectDrawBuffer {\n"
            "    RectDrawCommand commands[];\n"
            "};\n"
            "layout(push_constant) uniform Constants {\n"
            "    mat4 projView;\n"
            "} pcs;\n"
            "layout(location = 0) out vec4 outColor;\n"
            "layout(location = 1) in flat uint InstanceIndex;\n"
            "void main() {\n"
            "    outColor = commands[InstanceIndex].albedo;\n"
            "}\n";
        if(!vkCompileShader(device, fragmentShaderSrc, shaderc_fragment_shader,&fragmentShader)) return false;

    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount  = 1;
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

        if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &rectDescriptorSetLayout) != VK_SUCCESS) return false;
    }

    if(!vkCreateGraphicPipeline(
        vertexShader, fragmentShader,
        &rectPipeline, &rectPipelineLayout, outFormat,
        .pushConstantsSize = sizeof(pushConstants),
        .descriptorSetLayoutCount = 1,
        .descriptorSetLayouts = &rectDescriptorSetLayout,
    )) return false;

    return true;
}

#define MAX_TEXT_COUNT 128
static VkPipeline textPipeline;
static VkPipelineLayout textPipelineLayout;
static VkDescriptorSet textImageDescriptorSet = {0};
static VkDescriptorSetLayout textImageDescriptorSetLayout = {0};
static VkDescriptorSetLayout textDescriptorSetLayout = {0};
static VkImage textImage;
static VkImageView textImageView;
static VkDeviceMemory textImageMemory;
static size_t textImageStride;

static bool dd_init_text(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
    //TODO use precompiled shaders
    VkShaderModule vertexShader;
    const char* vertexShaderSrc =
            "#version 450\n"
            "#extension GL_EXT_scalar_block_layout : require\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "struct TextDrawCommand {\n"
            "    vec2 position;\n"
            "    vec2 scale;\n"
            "    vec2 uv_offset;\n"
            "    vec2 uv_size;\n"
            "    vec4 albedo;\n"
            "};\n"
            "layout(set = 0, binding = 0, scalar) readonly buffer TextDrawBuffer {\n"
            "    TextDrawCommand commands[];\n"
            "};\n"
            "layout(push_constant) uniform Constants {\n"
            "    mat4 projView;\n"
            "} pcs;\n"
            "layout(location = 0) out vec2 FragUV;\n"
            "layout(location = 1) out flat uint InstanceIndex;\n"
            "void main() {\n"
            "    uint b = 1 << (gl_VertexIndex % 6);\n"
            "    vec2 baseCoord = vec2((0x1C & b) != 0, (0xE & b) != 0);\n"
            "    TextDrawCommand cmd = commands[gl_InstanceIndex];\n"
            "    vec2 pos = cmd.position;\n"
            "    vec2 scale = cmd.scale;\n"
            "    mat4 model = mat4(\n"
            "        vec4(scale.x, 0,       0, 0),\n"
            "        vec4(0,       scale.y, 0, 0),\n"
            "        vec4(0,       0,       1, 0),\n"
            "        vec4(pos.x,   pos.y,   0, 1)\n"
            "    );\n"
            "    gl_Position = pcs.projView * model * vec4(baseCoord, 0.0, 1.0);\n"
            "    FragUV = cmd.uv_offset + baseCoord * cmd.uv_size;\n"
            "    InstanceIndex = gl_InstanceIndex;\n"
            "}\n";

    if(!vkCompileShader(device, vertexShaderSrc, shaderc_vertex_shader, &vertexShader)) return false;

    VkShaderModule fragmentShader;
    const char* fragmentShaderSrc =
            "#version 450\n"
            "#extension GL_EXT_scalar_block_layout : require\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "struct TextDrawCommand {\n"
            "    vec2 position;\n"
            "    vec2 scale;\n"
            "    vec2 uv_offset;\n"
            "    vec2 uv_size;\n"
            "    vec4 albedo;\n"
            "};\n"
            "layout(set = 0, binding = 0, scalar) readonly buffer TextDrawBuffer {\n"
            "    TextDrawCommand commands[];\n"
            "};\n"
            "layout(set = 1, binding = 0) uniform sampler2D fontAtlas;\n"
            "layout(push_constant) uniform Constants {\n"
            "    mat4 projView;\n"
            "} pcs;\n"
            "layout(location = 0) out vec4 outColor;\n"
            "layout(location = 0) in vec2 FragUV;\n"
            "layout(location = 1) in flat uint InstanceIndex;\n"
            "void main() {\n"
            "    TextDrawCommand cmd = commands[InstanceIndex];\n"
            "    float alpha = texture(fontAtlas, FragUV).r;\n"
            "    outColor = vec4(cmd.albedo.rgb, cmd.albedo.a * alpha);\n"
            "}\n";

    if(!vkCompileShader(device, fragmentShaderSrc, shaderc_fragment_shader, &fragmentShader)) return false;

    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount  = 1;
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

        if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &textDescriptorSetLayout) != VK_SUCCESS) return false;
    }

    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount  = 1;
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

        if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &textImageDescriptorSetLayout) != VK_SUCCESS) return false;
    }

    if(!vkCreateGraphicPipeline(
        vertexShader, fragmentShader,
        &textPipeline, &textPipelineLayout, outFormat,
        .pushConstantsSize = sizeof(pushConstants),
        .descriptorSetLayoutCount = 2,
        .descriptorSetLayouts = ((VkDescriptorSetLayout*)&(VkDescriptorSetLayout[]){textDescriptorSetLayout,textImageDescriptorSetLayout}),
    )) return false;

    int w,h;
    uint8_t* data = stbi_load("assets/font.png",&w,&h,NULL,4);
    if(!data) return false;

    void* textImageMapped;
    
    if(!dd_createMyImage(device, &textImage,
        w,h,
        &textImageMemory,
        &textImageView,
        &textImageStride,
        &textImageMapped,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    )) return false;

    for(size_t y = 0; y < h; y++){
        memcpy((uint8_t*)textImageMapped + textImageStride*y,
               (uint8_t*)data + w*sizeof(uint32_t)*y,
               w*sizeof(uint32_t)
            );
    }

    vkUnmapMemory(device, textImageMemory);

    stbi_image_free(data);

    dd_transitionMyImage(textImage,
        VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    if(!dd_create_image_descriptor_set_and_bind_image(device, descriptorPool, textImageDescriptorSetLayout, &textImageDescriptorSet, textImageView, dd_samplerNearest)) return false;

    return true;
}

#define MAX_IMAGE_COUNT 128
#define MAX_TEXTURES_COUNT 128
static VkPipeline imagesPipeline;
static VkPipelineLayout imagesPipelineLayout;
static VkDescriptorSet imagesImagesDescriptorSet = {0};
static VkDescriptorSetLayout imagesImagesDescriptorSetLayout = {0};
static VkDescriptorSetLayout imagesDescriptorSetLayout = {0};

static VkImage imagesNullTexture;
static VkImageView imagesNullTextureView;
static VkDeviceMemory imagesNullTextureMemory;
static size_t imagesNullTextureStride;
static size_t imagesNullTextureWidth = 8;
static size_t imagesNullTextureHeight = 8;

static bool dd_init_images(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
    //TODO use precompiled shaders
    VkShaderModule vertexShader;
    const char* vertexShaderSrc =
            "#version 450\n"
            "#extension GL_EXT_scalar_block_layout : require\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "struct ImageDrawCommand {\n"
            "    vec2 position;\n"
            "    vec2 scale;\n"
            "    vec2 uv_offset;\n"
            "    vec2 uv_size;\n"
            "    vec4 albedo;\n"
            "    uint texture_id;\n"
            "    uint _pad0;\n"
            "    uint _pad1;\n"
            "    uint _pad2;\n"
            "};\n"
            "layout(set = 0, binding = 0, scalar) readonly buffer TextDrawBuffer {\n"
            "    ImageDrawCommand commands[];\n"
            "};\n"
            "layout(push_constant) uniform Constants {\n"
            "    mat4 projView;\n"
            "} pcs;\n"
            "layout(location = 0) out vec2 FragUV;\n"
            "layout(location = 1) out flat uint InstanceIndex;\n"
            "void main() {\n"
            "    uint b = 1 << (gl_VertexIndex % 6);\n"
            "    vec2 baseCoord = vec2((0x1C & b) != 0, (0xE & b) != 0);\n"
            "    ImageDrawCommand cmd = commands[gl_InstanceIndex];\n"
            "    vec2 pos = cmd.position;\n"
            "    vec2 scale = cmd.scale;\n"
            "    mat4 model = mat4(\n"
            "        vec4(scale.x, 0,       0, 0),\n"
            "        vec4(0,       scale.y, 0, 0),\n"
            "        vec4(0,       0,       1, 0),\n"
            "        vec4(pos.x,   pos.y,   0, 1)\n"
            "    );\n"
            "    gl_Position = pcs.projView * model * vec4(baseCoord, 0.0, 1.0);\n"
            "    FragUV = cmd.uv_offset + baseCoord * cmd.uv_size;\n"
            "    InstanceIndex = gl_InstanceIndex;\n"
            "}\n";

    if(!vkCompileShader(device, vertexShaderSrc, shaderc_vertex_shader, &vertexShader)) return false;

    VkShaderModule fragmentShader;
    const char* fragmentShaderSrc =
            "#version 450\n"
            "#extension GL_EXT_scalar_block_layout : require\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "struct ImageDrawCommand {\n"
            "    vec2 position;\n"
            "    vec2 scale;\n"
            "    vec2 uv_offset;\n"
            "    vec2 uv_size;\n"
            "    vec4 albedo;\n"
            "    uint texture_id;\n"
            "    uint _pad0;\n"
            "    uint _pad1;\n"
            "    uint _pad2;\n"
            "};\n"
            "layout(set = 0, binding = 0, scalar) readonly buffer TextDrawBuffer {\n"
            "    ImageDrawCommand commands[];\n"
            "};\n"
            "layout (set = 1, binding = 0) uniform sampler2D textures[];\n"
            "layout(push_constant) uniform Constants {\n"
            "    mat4 projView;\n"
            "} pcs;\n"
            "layout(location = 0) out vec4 outColor;\n"
            "layout(location = 0) in vec2 FragUV;\n"
            "layout(location = 1) in flat uint InstanceIndex;\n"
            "void main() {\n"
            "    ImageDrawCommand cmd = commands[InstanceIndex];\n"
            "    outColor = texture(textures[cmd.texture_id], FragUV) * cmd.albedo;\n"
            "}\n";

    if(!vkCompileShader(device, fragmentShaderSrc, shaderc_fragment_shader, &fragmentShader)) return false;

    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.bindingCount  = 1;
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;

        if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &imagesDescriptorSetLayout) != VK_SUCCESS) return false;
    }

    {
        VkDescriptorBindingFlags descriptorBindingFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT | VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT;

        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptorSetLayoutBinding.descriptorCount = MAX_TEXTURES_COUNT;
        descriptorSetLayoutBinding.binding = 0;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_ALL;
        descriptorSetLayoutBinding.pImmutableSamplers = NULL;

        VkDescriptorSetLayoutBindingFlagsCreateInfo descriptorSetLayoutBindingFlagsCreateInfo = {0};
        descriptorSetLayoutBindingFlagsCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO;
        descriptorSetLayoutBindingFlagsCreateInfo.pNext = NULL;
        descriptorSetLayoutBindingFlagsCreateInfo.bindingCount = 1;
        descriptorSetLayoutBindingFlagsCreateInfo.pBindingFlags = &descriptorBindingFlags;

        VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};
        descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptorSetLayoutCreateInfo.pNext = &descriptorSetLayoutBindingFlagsCreateInfo;
        descriptorSetLayoutCreateInfo.bindingCount = 1;
        descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;
        descriptorSetLayoutCreateInfo.flags = 0;

        if(vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, NULL, &imagesImagesDescriptorSetLayout) != VK_SUCCESS) return false;
    }

    if(!vkCreateGraphicPipeline(
        vertexShader, fragmentShader,
        &imagesPipeline, &imagesPipelineLayout, outFormat,
        .pushConstantsSize = sizeof(pushConstants),
        .descriptorSetLayoutCount = 2,
        .descriptorSetLayouts = ((VkDescriptorSetLayout*)&(VkDescriptorSetLayout[]){imagesDescriptorSetLayout,imagesImagesDescriptorSetLayout}),
    )) return false;

    VkDescriptorSetVariableDescriptorCountAllocateInfo count_info = {0};
    count_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO;
    count_info.descriptorSetCount = 1;
    uint32_t max_binding = MAX_TEXTURES_COUNT;
    count_info.pDescriptorCounts = &max_binding;

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {0}; 
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.pNext = &count_info;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &imagesImagesDescriptorSetLayout;

    if(vkAllocateDescriptorSets(device, &descriptorSetAllocateInfo, &imagesImagesDescriptorSet) != VK_SUCCESS) return false;

    //creating null texture
    {
        uint32_t* nullTextureData = calloc(imagesNullTextureWidth*imagesNullTextureHeight,sizeof(uint32_t));

        for (size_t y = 0; y < imagesNullTextureHeight; y++) {
            for (size_t x = 0; x < imagesNullTextureWidth; x++) {
                size_t i = y * imagesNullTextureWidth + x;
                bool checker = (x + y) % 2; // alternate per row and column
                nullTextureData[i] = checker ? 0xFF000000 : 0xFFFF00FF;
            }
        }

        void* mapped;
        if(!dd_createMyImage(device, 
            &imagesNullTexture, 
            imagesNullTextureWidth, imagesNullTextureHeight, 
            &imagesNullTextureMemory, 
            &imagesNullTextureView, 
            &imagesNullTextureStride, 
            &mapped,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
        )) return false;

        for(size_t y = 0; y < imagesNullTextureHeight; y++){
            memcpy((uint8_t*)mapped + imagesNullTextureStride*y,
                   (uint8_t*)nullTextureData + sizeof(uint32_t)*imagesNullTextureWidth*y,
                   sizeof(uint32_t)*imagesNullTextureWidth);
        }

        free(nullTextureData);

        vkUnmapMemory(device, imagesNullTextureMemory);

        dd_transitionMyImage(imagesNullTexture,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

        VkDescriptorImageInfo*  descriptorImageInfos = (VkDescriptorImageInfo*)calloc(MAX_TEXTURES_COUNT, sizeof(VkDescriptorImageInfo));
        for(size_t i = 0; i < MAX_TEXTURES_COUNT; i++){
            descriptorImageInfos[i] = (VkDescriptorImageInfo){
                .sampler = dd_samplerNearest,
                .imageView = imagesNullTextureView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
        }

        vkUpdateDescriptorSets(device,1,&(VkWriteDescriptorSet){
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = NULL,
            .dstSet = imagesImagesDescriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = MAX_TEXTURES_COUNT,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = descriptorImageInfos,
        },0,NULL);

        free(descriptorImageInfos);
    }

    return true;
}

static VkDescriptorPool uiDescriptorPool;
static VkDevice uiDevice;
bool dd_init(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
    if(inited) return true;

    if(vkCreateSampler(device, &(VkSamplerCreateInfo){
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_NEAREST,
        .minFilter = VK_FILTER_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 1.0f,
        .borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
        .unnormalizedCoordinates = VK_FALSE,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .mipLodBias = 0.0f,
        .minLod = 0.0f,
        .maxLod = VK_LOD_CLAMP_NONE,
    }, NULL, &dd_samplerNearest) != VK_SUCCESS) return false;

    if(!dd_init_rects(device, outFormat, descriptorPool)) return false;
    if(!dd_init_text(device, outFormat, descriptorPool)) return false;
    if(!dd_init_images(device, outFormat, descriptorPool)) return false;
    uiDescriptorPool = descriptorPool;
    uiDevice = device;

    inited = true;
    return true;
}

void dd_begin(){
    drawCommands.count = 0;
    drawCommandsArena.size = 0;
    return;
}

void dd_end(){
    return;
}

static void dd_cmds_push(DrawCommands* drawCommands, DrawCommand drawCommand){
    assert(drawCommand.type != DD_DRAW_CMD_NONE && drawCommand.type != DD_DRAW_CMDS_COUNT);

    if(drawCommand.type == DD_DRAW_CMD_RECT) da_arena_push(&drawCommandsArena, drawCommand.as.rect);
    else if(drawCommand.type == DD_DRAW_CMD_SCISSOR) da_arena_push(&drawCommandsArena, drawCommand.as.scissor);
    else if(drawCommand.type == DD_DRAW_CMD_TEXT) da_arena_push(&drawCommandsArena, drawCommand.as.text);
    else if(drawCommand.type == DD_DRAW_CMD_IMAGE) da_arena_push(&drawCommandsArena, drawCommand.as.image);
    else assert(false && "Implement this type");

    da_push(drawCommands, drawCommand.type);
}

void dd_rect(float x, float y, float w, float h, uint32_t color){
    dd_cmds_push(&drawCommands, ((DrawCommand){
        .type = DD_DRAW_CMD_RECT,
        .as.rect = (RectDrawCommand){
            .position.x = x,
            .position.y = y,
            .scale.x = w,
            .scale.y = h,
            .albedo = {
                .x = ((float)(((color) >> 16) & 0xFF) / 255.0f),
                .y = ((float)(((color) >>  8) & 0xFF) / 255.0f),
                .z = ((float)(((color) >>  0) & 0xFF) / 255.0f),
                .w = ((float)(((color) >> 24) & 0xFF) / 255.0f), 
            }
        },
    }));
}

#define TEXT_WIDTH_RATIO (0.5)

void dd_text(const char* text, float x, float y, float size, uint32_t color){
    if(!text) return;
    size_t n = strlen(text);
    float origin_x = 0;
    float origin_y = 0;
    for(size_t i = 0; i < n; i++){
        size_t ch = (size_t)text[i];
        if(ch == '\n'){
            origin_y += size;
            origin_x = 0;
            continue;
        }
        dd_cmds_push(&drawCommands, ((DrawCommand){
        .type = DD_DRAW_CMD_TEXT,
            .as.text = (TextDrawCommand){
                .position.x = origin_x + x,
                .position.y = origin_y + y,
                .scale.x = size,
                .scale.y = size,
                .albedo = {
                    .x = ((float)(((color) >> 16) & 0xFF) / 255.0f),
                    .y = ((float)(((color) >>  8) & 0xFF) / 255.0f),
                    .z = ((float)(((color) >>  0) & 0xFF) / 255.0f),
                    .w = ((float)(((color) >> 24) & 0xFF) / 255.0f), 
                },
                .uv_offset.x = 1.0 / 16 * (double)(ch % 16),
                .uv_offset.y = 1.0 / 16 * (double)(ch / 16),
                .uv_size.x = 1.0 / 16,
                .uv_size.y = 1.0 / 16,
            },
        }));
        origin_x += size*TEXT_WIDTH_RATIO;
    }
}

float dd_text_measure(const char* text, float size){
    float out = 0;
    float currentLine = 0;
    size_t n = strlen(text);
    for(size_t i = 0; i < n; i++){
        size_t ch = (size_t)text[i];
        if(ch == '\n'){
            if(currentLine > out) out = currentLine;
            currentLine = 0;
            continue;
        }
        currentLine += size*TEXT_WIDTH_RATIO;
    }

    if(currentLine > out) out = currentLine;

    return out;
}

typedef struct{
    VkImage image;
    VkImageView view;
    VkDeviceMemory memory;
    size_t stride;
    size_t width;
    size_t height;
    bool used;
} DDTexture;

typedef struct{
    DDTexture* items;
    size_t count;
    size_t capacity;
} DDTexturePool;

uint32_t DDTexturePool_find_empty(DDTexturePool* pool){
    if(pool->capacity == 0) {
        da_reserve(pool, pool->count);
        for(size_t i = 0; i < pool->count; i++){
            pool->items[i] = (DDTexture){0};
        }
    }

    for(size_t i = 0; i < pool->count; i++){
        if(pool->items[i].used == false) return i;
    }
    return -1;
}

static DDTexturePool uITexturePool = {.count = MAX_TEXTURES_COUNT};

void dd_image(uint32_t texture_id, float x, float y, float w, float h, float uv_x, float uv_y, float uv_w, float uv_h, uint32_t albedo){
    if(texture_id >= uITexturePool.count) return;
    dd_cmds_push(&drawCommands, ((DrawCommand){
        .type = DD_DRAW_CMD_IMAGE,
        .as.image = (ImageDrawCommand){
            .texture_id = texture_id,

            .position.x = x,
            .position.y = y,
            .scale.x = w,
            .scale.y = h,
            .albedo = {
                .x = ((float)(((albedo) >> 16) & 0xFF) / 255.0f),
                .y = ((float)(((albedo) >>  8) & 0xFF) / 255.0f),
                .z = ((float)(((albedo) >>  0) & 0xFF) / 255.0f),
                .w = ((float)(((albedo) >> 24) & 0xFF) / 255.0f), 
            },

            .uv_offset.x = uv_x,
            .uv_offset.y = uv_y,
            .uv_size.x = uv_w,
            .uv_size.y = uv_h,
        },
    }));
}

uint32_t dd_create_texture(size_t width, size_t height){
    uint32_t texture_id = DDTexturePool_find_empty(&uITexturePool);
    if(texture_id == -1) return texture_id; // couldnt find empty texture (pool full)

    DDTexture* texture = &uITexturePool.items[texture_id];
    
    if(!dd_createMyImage(uiDevice, 
        &texture->image, 
        width, height, 
        &texture->memory, 
        &texture->view, 
        &texture->stride, 
        NULL, 
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
    )) return -1;

    dd_transitionMyImage(texture->image,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    vkUpdateDescriptorSets(uiDevice,1,&(VkWriteDescriptorSet){
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext = NULL,
        .dstSet = imagesImagesDescriptorSet,
        .dstBinding = 0,
        .dstArrayElement = texture_id,
        .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo = &(VkDescriptorImageInfo){
            .sampler = dd_samplerNearest,
            .imageView = texture->view,
            .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        },
    },0,NULL);

    texture->width = width;
    texture->height = height;
    texture->used = true;
    return texture_id;
}

bool dd_update_texture(uint32_t texture_id, void* data){
    if(texture_id == -1) return false;
    if(texture_id >= uITexturePool.count) return false;

    DDTexture* texture = &uITexturePool.items[texture_id];

    void *mapped;
    if(vkMapMemory(uiDevice, texture->memory, 0, texture->stride*texture->height, 0, &mapped) != VK_SUCCESS) return false;

    for(size_t y = 0; y < texture->height; y++){
        memcpy((uint8_t*)mapped + texture->stride*y,
               (uint8_t*)data + sizeof(uint32_t)*texture->width*y,
               sizeof(uint32_t)*texture->width
            );
    }

    vkUnmapMemory(uiDevice, texture->memory);
    return true;
}

bool dd_destroy_texture(uint32_t texture_id) {
    if (texture_id == (uint32_t)-1) return false;
    if (texture_id >= uITexturePool.count) return false;

    DDTexture* texture = &uITexturePool.items[texture_id];
    if (!texture->used) return false;

    if (texture->view != VK_NULL_HANDLE) {
        vkDestroyImageView(uiDevice, texture->view, NULL);
        texture->view = VK_NULL_HANDLE;
    }

    if (texture->image != VK_NULL_HANDLE) {
        vkDestroyImage(uiDevice, texture->image, NULL);
        texture->image = VK_NULL_HANDLE;
    }

    if (texture->memory != VK_NULL_HANDLE) {
        vkFreeMemory(uiDevice, texture->memory, NULL);
        texture->memory = VK_NULL_HANDLE;
    }

    VkDescriptorImageInfo nullImageInfo = {
        .sampler     = dd_samplerNearest,
        .imageView   = imagesNullTextureView,
        .imageLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    VkWriteDescriptorSet write = {
        .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .pNext           = NULL,
        .dstSet          = imagesImagesDescriptorSet,
        .dstBinding      = 0,
        .dstArrayElement = texture_id,
        .descriptorCount = 1,
        .descriptorType  = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        .pImageInfo      = &nullImageInfo,
    };

    vkUpdateDescriptorSets(uiDevice, 1, &write, 0, NULL);

    texture->stride = 0;
    texture->width = 0;
    texture->height = 0;
    texture->used = false;

    return true;
}

void dd_scissor(float x, float y, float w, float h){
    dd_cmds_push(&drawCommands, ((DrawCommand){
        .type = DD_DRAW_CMD_SCISSOR,
        .as.scissor = (ScissorDrawCommand){
            .offset.x = x,
            .offset.y = y,
            .size.x = w,
            .size.y = h,
        },
    }));
}

// drawing
typedef struct{
    VkBuffer buffer;
    VkDeviceMemory memory;
    void* mapped;
    VkDescriptorSet descriptorSet;
    size_t size;
} DDBuffer;

typedef struct{
    DDBuffer* items;
    size_t count;
    size_t capacity;
    size_t used;
    size_t item_size;
} DDBufferPool;

DDBuffer* DDBufferPool_get_avaliable(DDBufferPool* pool, VkDevice device, VkDescriptorPool descriptorPool){
    if(pool->used < pool->count) return &pool->items[pool->used++];
    da_reserve(pool, pool->count + 1);
    DDBuffer* out = &pool->items[pool->count++];

    //initalization
    VkDeviceSize bufferSize = pool->item_size;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,bufferSize,&out->buffer,&out->memory)) return false;
    if(vkMapMemory(device, out->memory, 0, bufferSize, 0, &out->mapped) != VK_SUCCESS) return false;
    if(!dd_create_buffer_descriptor_set_and_bind_buffer(device, descriptorPool, rectDescriptorSetLayout, &out->descriptorSet, out->buffer, bufferSize)) return false;
    out->size = bufferSize;

    return out;
}

void DDBufferPool_reset(DDBufferPool* pool){
    pool->used = 0;
}

static void dd_draw_rects(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment, VkRect2D scissor, void* mapped, VkDescriptorSet descriptorSet, RectDrawCommand* rects_ptr, size_t rects_count){
    assert(rects_count <= MAX_RECT_COUNT);
    assert(mapped && descriptorSet && "Provide those");

    memcpy(mapped, rects_ptr, rects_count*sizeof(*rects_ptr));

    vkCmdBeginRenderingEX(cmd,
        .colorAttachment = colorAttachment,
        .clearBackground = false,
        .renderArea = (
            (VkExtent2D){.width = screenWidth, .height = screenHeight}
        )
    );

    vkCmdSetViewport(cmd, 0, 1, &(VkViewport){
        .width = screenWidth,
        .height = screenHeight
    });
        
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS, rectPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,rectPipelineLayout,0,1,&descriptorSet,0,NULL);
    vkCmdPushConstants(cmd, rectPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pushConstants);
    vkCmdDraw(cmd, 6, rects_count, 0, 0);
    vkCmdEndRendering(cmd);
}

static void dd_draw_text(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment, VkRect2D scissor, void* mapped, VkDescriptorSet descriptorSet, TextDrawCommand* text_ptr, size_t text_count){
    assert(text_count <= MAX_TEXT_COUNT);
    assert(textImageDescriptorSet && "Initialize this");
    assert(mapped && descriptorSet && "Provide those");

    memcpy(mapped, text_ptr, text_count*sizeof(*text_ptr));

    vkCmdBeginRenderingEX(cmd,
        .colorAttachment = colorAttachment,
        .clearBackground = false,
        .renderArea = (
            (VkExtent2D){.width = screenWidth, .height = screenHeight}
        )
    );

    vkCmdSetViewport(cmd, 0, 1, &(VkViewport){
        .width = screenWidth,
        .height = screenHeight
    });
        
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS, textPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,textPipelineLayout,0,2,(VkDescriptorSet*)&(VkDescriptorSet[]){descriptorSet, textImageDescriptorSet},0,NULL);
    vkCmdPushConstants(cmd, textPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pushConstants);
    vkCmdDraw(cmd, 6, text_count, 0, 0);
    vkCmdEndRendering(cmd);
}

static void dd_draw_images(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment, VkRect2D scissor, void* mapped, VkDescriptorSet descriptorSet, ImageDrawCommand* images_ptr, size_t images_count){
    assert(images_count <= MAX_IMAGE_COUNT);
    assert(imagesImagesDescriptorSet && "Initialize this");
    assert(mapped && descriptorSet && "Provide those");

    memcpy(mapped, images_ptr, images_count*sizeof(*images_ptr));

    vkCmdBeginRenderingEX(cmd,
        .colorAttachment = colorAttachment,
        .clearBackground = false,
        .renderArea = (
            (VkExtent2D){.width = screenWidth, .height = screenHeight}
        )
    );

    vkCmdSetViewport(cmd, 0, 1, &(VkViewport){
        .width = screenWidth,
        .height = screenHeight
    });
        
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS, imagesPipeline);
    vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_GRAPHICS,imagesPipelineLayout,0,2,(VkDescriptorSet*)&(VkDescriptorSet[]){descriptorSet, imagesImagesDescriptorSet},0,NULL);
    vkCmdPushConstants(cmd, imagesPipelineLayout, VK_SHADER_STAGE_ALL, 0, sizeof(PushConstants), &pushConstants);
    vkCmdDraw(cmd, 6, images_count, 0, 0);
    vkCmdEndRendering(cmd);
}

static DDBufferPool tempDDRectBufferPool = {.item_size = sizeof(RectDrawCommand)*MAX_RECT_COUNT};
static DDBufferPool tempDDTextBufferPool = {.item_size = sizeof(TextDrawCommand)*MAX_TEXT_COUNT};
static DDBufferPool tempDDImageBufferPool = {.item_size = sizeof(ImageDrawCommand)*MAX_IMAGE_COUNT};
size_t oldScreenWidth = 0;
size_t oldScreenHeight = 0;

void dd_draw(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment){
    if (drawCommands.count == 0) return;

    if (oldScreenWidth != screenWidth || oldScreenHeight != screenHeight) {
        mat4 ortho = ortho2D(screenWidth, screenHeight);
        oldScreenWidth = screenWidth;
        oldScreenHeight = screenHeight;
        pushConstants = (PushConstants){
            .projView = mat4mul(&ortho, &(mat4){
                1,0,0,0,
                0,1,0,0,
                0,0,1,0,
                -((float)screenWidth)/2, -((float)screenHeight)/2, 0, 1,
            }),
        };
    }

    size_t index = 0;
    DDBufferPool_reset(&tempDDRectBufferPool);
    DDBufferPool_reset(&tempDDTextBufferPool);
    DDBufferPool_reset(&tempDDImageBufferPool);

    VkRect2D scissor_default = {
        .offset.x = 0,
        .offset.y = 0,
        .extent.width = screenWidth,
        .extent.height = screenHeight
    };

    VkRect2D scissor = scissor_default;
    VkRect2D new_scissor = scissor;
    bool scissor_changed = false;

    void* ptr = drawCommandsArena.buff;

    while (index < drawCommands.count) {
        UiDrawCmdType type = DD_DRAW_CMD_NONE;

        size_t n = 0;
        size_t to_push = 0;

        while (index < drawCommands.count) {
            UiDrawCmdType cur_type = drawCommands.items[index];
            if (type == DD_DRAW_CMD_NONE) type = cur_type;
            else if (type != cur_type) break;

            if(type == DD_DRAW_CMD_SCISSOR) {
                ScissorDrawCommand* cur_as_scissor = (ScissorDrawCommand*)ptr;
                if(cur_as_scissor->size.x == 0 && cur_as_scissor->size.y == 0){
                    new_scissor = scissor_default;
                }else{
                    new_scissor = (VkRect2D){
                        .offset.x = cur_as_scissor->offset.x,
                        .offset.y = cur_as_scissor->offset.y,
                        .extent.width = cur_as_scissor->size.x,
                        .extent.height = cur_as_scissor->size.y,
                    };
                }

                scissor_changed = true;
            }
            else if(type == DD_DRAW_CMD_NONE) assert(false && "Unreachable NONE");
            else if(type == DD_DRAW_CMDS_COUNT) assert(false && "Unreachable COUNT");
            n++;
            index++;

            if(type == DD_DRAW_CMD_SCISSOR) to_push += sizeof(ScissorDrawCommand);
            else if(type == DD_DRAW_CMD_RECT) to_push += sizeof(RectDrawCommand);
            else if(type == DD_DRAW_CMD_TEXT) to_push += sizeof(TextDrawCommand);
            else if(type == DD_DRAW_CMD_IMAGE) to_push += sizeof(ImageDrawCommand);
            else assert(false && "Unreachable TYPE");

            if(type == DD_DRAW_CMD_RECT && n >= MAX_RECT_COUNT) break;
            else if(type == DD_DRAW_CMD_TEXT && n >= MAX_TEXT_COUNT) break;
            else if(type == DD_DRAW_CMD_IMAGE && n >= MAX_IMAGE_COUNT) break;
        }

        if(!(scissor.extent.width == 0 || scissor.extent.height == 0)) {
            //drawing
            if (type == DD_DRAW_CMD_RECT) {
                assert(n <= sizeof(RectDrawCommand)*MAX_RECT_COUNT);
                DDBuffer* buff = DDBufferPool_get_avaliable(&tempDDRectBufferPool, uiDevice, uiDescriptorPool);
                assert(buff->size == sizeof(RectDrawCommand)*MAX_RECT_COUNT);
                dd_draw_rects(cmd, screenWidth, screenHeight, colorAttachment, scissor, buff->mapped,buff->descriptorSet, (RectDrawCommand*)ptr, n);
            }else if(type == DD_DRAW_CMD_TEXT){
                assert(n <= sizeof(TextDrawCommand)*MAX_RECT_COUNT);
                DDBuffer* buff = DDBufferPool_get_avaliable(&tempDDTextBufferPool, uiDevice, uiDescriptorPool);
                assert(buff->size == sizeof(TextDrawCommand)*MAX_TEXT_COUNT);
                dd_draw_text(cmd, screenWidth, screenHeight, colorAttachment, scissor, buff->mapped, buff->descriptorSet, (TextDrawCommand*)ptr, n);
            }else if(type == DD_DRAW_CMD_IMAGE){
                assert(n <= sizeof(ImageDrawCommand)*MAX_RECT_COUNT);
                DDBuffer* buff = DDBufferPool_get_avaliable(&tempDDImageBufferPool, uiDevice, uiDescriptorPool);
                assert(buff->size == sizeof(ImageDrawCommand)*MAX_IMAGE_COUNT);
                dd_draw_images(cmd, screenWidth, screenHeight, colorAttachment, scissor, buff->mapped, buff->descriptorSet, (ImageDrawCommand*)ptr, n);
            }
        }
        
        if(scissor_changed){
            scissor_changed = false;
            scissor = new_scissor;
        }

        ptr += to_push;
    }
}
