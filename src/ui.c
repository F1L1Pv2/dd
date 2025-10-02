#include "ui.h"

#include "vulkan/vulkan.h"
#include "engine/vulkan_createGraphicPipelines.h"
#include "engine/vulkan_compileShader.h"
#include "engine/vulkan_helpers.h"
#include "engine/vulkan_buffer.h"
#include "engine/vulkan_images.h"
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include "thirdparty/stb_image.h"

#define DA_INIT_CAP 256

#define da_reserve(da, expected_capacity)                                                  \
    do {                                                                                   \
        if ((expected_capacity) > (da)->capacity) {                                        \
            if ((da)->capacity == 0) {                                                     \
                (da)->capacity = DA_INIT_CAP;                                              \
            }                                                                              \
            while ((expected_capacity) > (da)->capacity) {                                 \
                (da)->capacity *= 2;                                                       \
            }                                                                              \
            (da)->items = realloc((da)->items, (da)->capacity * sizeof(*(da)->items));     \
            assert((da)->items != NULL && "Buy more RAM lol");                             \
        }                                                                                  \
    } while (0)

// Append an item to a dynamic array
#define da_append(da, item)                    \
    do {                                       \
        da_reserve((da), (da)->count + 1); \
        (da)->items[(da)->count++] = (item);   \
    } while (0)

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

mat4 ortho2D(float width, float height){
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

mat4 mat4mul(mat4 *a, mat4 *b) {
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

typedef struct{
    RectDrawCommand* items;
    size_t count;
    size_t capacity;
} RectDrawCommands;

typedef struct {
    vec2 position;
    vec2 scale;
    vec2 uv_offset;
    vec2 uv_size;
    vec4 albedo;
} TextDrawCommand;

typedef struct{
    TextDrawCommand* items;
    size_t count;
    size_t capacity;
} TextDrawCommands;

typedef enum {
    UI_DRAW_CMD_NONE = 0,
    UI_DRAW_CMD_RECT,
    UI_DRAW_CMD_SCISSOR,
    UI_DRAW_CMD_TEXT,
    UI_DRAW_CMDS_COUNT,
} UiDrawCmdType;

typedef struct{
    UiDrawCmdType type;
    union{
        RectDrawCommand rect;
        ScissorDrawCommand scissor;
        TextDrawCommand text;
    } as;
} DrawCommand;

typedef struct{
    DrawCommand* items;
    size_t count;
    size_t capacity;
} DrawCommands;

typedef struct{
    mat4 projView;
} PushConstants;

static bool inited = false;

static VkPipeline rectPipeline;
static VkPipelineLayout rectPipelineLayout;
static VkDescriptorSetLayout rectDescriptorSetLayout = {0};

static VkPipeline textPipeline;
static VkPipelineLayout textPipelineLayout;
static VkDescriptorSet textImageDescriptorSet = {0};
static VkDescriptorSetLayout textImageDescriptorSetLayout = {0};
static VkDescriptorSetLayout textDescriptorSetLayout = {0};
static VkImage textImage;
static VkImageView textImageView;
static VkDeviceMemory textImageMemory;
static size_t textImageStride;
static void* textImageMapped;

static VkSampler ui_samplerNearest;

static DrawCommands drawCommands = {0};

#define MAX_RECT_COUNT 128
#define MAX_TEXT_COUNT 128
static PushConstants pushConstants;

static bool ui_create_buffer_descriptor_set_and_bind_buffer(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet* descriptorSetOut, VkBuffer buffer, VkDeviceSize size){
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

static bool ui_create_image_descriptor_set_and_bind_image(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSetLayout descriptorSetLayout, VkDescriptorSet* descriptorSetOut, VkImageView imageView, VkSampler sampler){
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

static bool ui_init_rects(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
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

static void ui_transitionMyImage_inner(VkCommandBuffer tempCmd, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlagBits oldStage, VkPipelineStageFlagBits newStage){
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

static void ui_transitionMyImage(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkPipelineStageFlagBits oldStage, VkPipelineStageFlagBits newStage){
    VkCommandBuffer tempCmd = vkCmdBeginSingleTime();
    ui_transitionMyImage_inner(tempCmd, image, oldLayout, newLayout, oldStage, newStage);
    vkCmdEndSingleTime(tempCmd);
}

static bool ui_createMyImage(VkDevice device, VkImage* image, size_t width, size_t height, VkDeviceMemory* imageMemory, VkImageView* imageView, size_t* imageStride, void** imageMapped, VkImageUsageFlagBits imageUsage, VkMemoryPropertyFlagBits memoryProperty){
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
    vkMapMemory(device,*imageMemory, 0, (*imageStride)*height, 0, imageMapped);

    return true;
}

static bool ui_init_text(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
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

    //init image

    //ui_create_image_descriptor_set_and_bind_image

    int w,h;
    uint8_t* data = stbi_load("assets/font.png",&w,&h,NULL,4);
    if(!data) return false;
    
    if(!ui_createMyImage(device, &textImage,
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

    stbi_image_free(data);

    ui_transitionMyImage(textImage,
        VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

    {
        VkSamplerCreateInfo samplerCreateInfo = {0};
        samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerCreateInfo.magFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.minFilter = VK_FILTER_NEAREST;
        samplerCreateInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        samplerCreateInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        samplerCreateInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
        samplerCreateInfo.anisotropyEnable = VK_FALSE;
        samplerCreateInfo.maxAnisotropy = 1.0f;
        samplerCreateInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerCreateInfo.unnormalizedCoordinates = VK_FALSE;
        samplerCreateInfo.compareEnable = VK_FALSE;
        samplerCreateInfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerCreateInfo.mipLodBias = 0.0f;
        samplerCreateInfo.minLod = 0.0f;
        samplerCreateInfo.maxLod = VK_LOD_CLAMP_NONE;

        vkCreateSampler(device, &samplerCreateInfo, NULL, &ui_samplerNearest);
    }

    if(!ui_create_image_descriptor_set_and_bind_image(device, descriptorPool, textImageDescriptorSetLayout, &textImageDescriptorSet, textImageView, ui_samplerNearest)) return false;

    return true;
}

static VkDescriptorPool uiDescriptorPool;
static VkDevice uiDevice;
bool ui_init(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
    if(inited) return true;

    if(!ui_init_rects(device, outFormat, descriptorPool)) return false;
    if(!ui_init_text(device, outFormat, descriptorPool)) return false;
    uiDescriptorPool = descriptorPool;
    uiDevice = device;

    inited = true;
    return true;
}

void ui_begin(
    size_t mouseX, size_t mouseY, 
    bool mouse_left_down,
    bool mouse_left_justPressed,
    bool mouse_left_justReleased,

    bool mouse_middle_down,
    bool mouse_middle_justPressed,
    bool mouse_middle_justReleased,

    bool mouse_right_down,
    bool mouse_right_justPressed,
    bool mouse_right_justReleased,
    double mouse_scroll,

    char* lastTextKey // used for typing in text boxes
){
    drawCommands.count = 0;
    return;
}

void ui_end(){
    return;
}

void ui_rect(float x, float y, float w, float h, uint32_t color){
    da_append(&drawCommands, ((DrawCommand){
        .type = UI_DRAW_CMD_RECT,
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

void ui_text(const char* text, float x, float y, float size, uint32_t color){
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
        da_append(&drawCommands, ((DrawCommand){
        .type = UI_DRAW_CMD_TEXT,
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

float ui_text_measure(const char* text, float size){
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

void ui_scissor(float x, float y, float w, float h){
    da_append(&drawCommands, ((DrawCommand){
        .type = UI_DRAW_CMD_SCISSOR,
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
} UIBuffer;

typedef struct{
    UIBuffer* items;
    size_t count;
    size_t capacity;
    size_t used;
    size_t item_size;
} UIBufferPool;

UIBuffer* UIBufferPool_get_avaliable(UIBufferPool* pool, VkDevice device, VkDescriptorPool descriptorPool){
    if(pool->used < pool->count) return &pool->items[pool->used++];
    da_reserve(pool, pool->count + 1);
    UIBuffer* out = &pool->items[pool->count++];

    //initalization
    VkDeviceSize bufferSize = pool->item_size;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,bufferSize,&out->buffer,&out->memory)) return false;
    if(vkMapMemory(device, out->memory, 0, bufferSize, 0, &out->mapped) != VK_SUCCESS) return false;
    if(!ui_create_buffer_descriptor_set_and_bind_buffer(device, descriptorPool, rectDescriptorSetLayout, &out->descriptorSet, out->buffer, bufferSize)) return false;
    out->size = bufferSize;

    return out;
}

void UIBufferPool_reset(UIBufferPool* pool){
    pool->used = 0;
}

static void ui_draw_rects(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment, VkRect2D scissor, void* mapped, VkDescriptorSet descriptorSet, RectDrawCommands* rects){
    assert(rects->count <= MAX_RECT_COUNT);
    assert(mapped && descriptorSet && "Provide those");

    memcpy(mapped, rects->items, rects->count*sizeof(rects->items[0]));

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
    vkCmdDraw(cmd, 6, rects->count, 0, 0);
    vkCmdEndRendering(cmd);
}

static void ui_draw_text(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment, VkRect2D scissor, void* mapped, VkDescriptorSet descriptorSet, TextDrawCommands* text){
    assert(text->count <= MAX_TEXT_COUNT);
    assert(textImageDescriptorSet && "Initialize this");
    assert(mapped && descriptorSet && "Provide those");

    memcpy(mapped, text->items, text->count*sizeof(text->items[0]));

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
    vkCmdDraw(cmd, 6, text->count, 0, 0);
    vkCmdEndRendering(cmd);
}

static TextDrawCommands tempTextDrawCommands = {0};
static RectDrawCommands tempRectDrawCommands = {0};
static UIBufferPool tempUIRectBufferPool = {.item_size = sizeof(RectDrawCommand)*MAX_RECT_COUNT};
static UIBufferPool tempUITextBufferPool = {.item_size = sizeof(TextDrawCommand)*MAX_TEXT_COUNT};
size_t oldScreenWidth = 0;
size_t oldScreenHeight = 0;

void ui_draw(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment){
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
    tempRectDrawCommands.count = 0;
    UIBufferPool_reset(&tempUIRectBufferPool);

    tempTextDrawCommands.count = 0;
    UIBufferPool_reset(&tempUITextBufferPool);

    VkRect2D scissor_default = {
        .offset.x = 0,
        .offset.y = 0,
        .extent.width = screenWidth,
        .extent.height = screenHeight
    };

    VkRect2D scissor = scissor_default;
    VkRect2D new_scissor = scissor;
    bool scissor_changed = false;

    while (index < drawCommands.count) {
        UiDrawCmdType type = UI_DRAW_CMD_NONE;

        while (index < drawCommands.count) {
            DrawCommand* cur = &drawCommands.items[index];
            if(cur->type == UI_DRAW_CMD_SCISSOR){
                if(cur->as.scissor.size.x == 0 && cur->as.scissor.size.y == 0){
                    new_scissor = scissor_default;
                }else{
                    new_scissor = (VkRect2D){
                        .offset.x = cur->as.scissor.offset.x,
                        .offset.y = cur->as.scissor.offset.y,
                        .extent.width = cur->as.scissor.size.x,
                        .extent.height = cur->as.scissor.size.y,
                    };
                }

                scissor_changed = true;
                index++;
                break;
            }
            else if (type == UI_DRAW_CMD_NONE) type = cur->type;
            else if (type != cur->type) break;

            if (type == UI_DRAW_CMD_RECT) da_append(&tempRectDrawCommands, cur->as.rect);
            else if(type == UI_DRAW_CMD_TEXT) da_append(&tempTextDrawCommands, cur->as.text);
            else if(type == UI_DRAW_CMD_NONE) assert(false && "Unreachable NONE");
            else if(type == UI_DRAW_CMDS_COUNT) assert(false && "Unreachable COUNT");
            else assert(false && "Unreachable TYPE");

            index++;
            if (type == UI_DRAW_CMD_RECT && tempRectDrawCommands.count >= MAX_RECT_COUNT) break;
            else if (type == UI_DRAW_CMD_TEXT && tempTextDrawCommands.count >= MAX_TEXT_COUNT) break;
        }

        if(!(scissor.extent.width == 0 || scissor.extent.height == 0)) {
            //drawing
            if (type == UI_DRAW_CMD_RECT) {
                UIBuffer* buff = UIBufferPool_get_avaliable(&tempUIRectBufferPool, uiDevice, uiDescriptorPool);
                assert(buff->size == sizeof(RectDrawCommand)*MAX_RECT_COUNT);
                ui_draw_rects(cmd, screenWidth, screenHeight, colorAttachment, scissor, buff->mapped,buff->descriptorSet, &tempRectDrawCommands);
            }else if(type == UI_DRAW_CMD_TEXT){
                UIBuffer* buff = UIBufferPool_get_avaliable(&tempUITextBufferPool, uiDevice, uiDescriptorPool);
                assert(buff->size == sizeof(TextDrawCommand)*MAX_TEXT_COUNT);
                ui_draw_text(cmd, screenWidth, screenHeight, colorAttachment, scissor, buff->mapped, buff->descriptorSet, &tempTextDrawCommands);
            }
        }

        //cleanup
        if (type == UI_DRAW_CMD_RECT) tempRectDrawCommands.count = 0;
        else if(type == UI_DRAW_CMD_TEXT) tempTextDrawCommands.count = 0;
        
        if(scissor_changed){
            scissor_changed = false;
            scissor = new_scissor;
        }
    }
}
