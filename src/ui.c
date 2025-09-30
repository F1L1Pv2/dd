#include "ui.h"

#include "vulkan/vulkan.h"
#include "engine/vulkan_createGraphicPipelines.h"
#include "engine/vulkan_compileShader.h"
#include "engine/vulkan_helpers.h"
#include "engine/vulkan_buffer.h"
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

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

typedef enum {
    UI_DRAW_CMD_NONE = 0,
    UI_DRAW_CMD_RECT,
    UI_DRAW_CMD_SCISSOR,
    UI_DRAW_CMDS_COUNT,
} UiDrawCmdType;

typedef struct{
    UiDrawCmdType type;
    union{
        RectDrawCommand rect;
        ScissorDrawCommand scissor;
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

static DrawCommands drawCommands = {0};

#define MAX_RECT_COUNT 128
static PushConstants pushConstants;

static bool ui_create_rect_descriptor_set_and_bind_buffer(VkDevice device, VkDescriptorPool descriptorPool, VkDescriptorSet* descriptorSetOut, VkBuffer buffer){
    VkResult result;
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {0};
    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool = descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &rectDescriptorSetLayout;
    
    result = vkAllocateDescriptorSets(device,&descriptorSetAllocateInfo, descriptorSetOut);
    if(result != VK_SUCCESS) return false;

    VkDescriptorBufferInfo descriptorBufferInfo = {
        .buffer = buffer,
        .offset = 0,
        .range = sizeof(RectDrawCommand)*MAX_RECT_COUNT,
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

static VkDescriptorPool uiDescriptorPool;
static VkDevice uiDevice;
bool ui_init(VkDevice device, VkFormat outFormat, VkDescriptorPool descriptorPool){
    if(inited) return true;

    if(!ui_init_rects(device, outFormat, descriptorPool)) return false;
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
} UIRectBuffer;

typedef struct{
    UIRectBuffer* items;
    size_t count;
    size_t capacity;
    size_t used;
} UIRectBufferPool;

UIRectBuffer* UIRectBufferPool_get_avaliable(UIRectBufferPool* pool, VkDevice device, VkDescriptorPool descriptorPool){
    if(pool->used < pool->count) return &pool->items[pool->used++];
    da_reserve(pool, pool->count + 1);
    UIRectBuffer* out = &pool->items[pool->count++];

    //initalization
    VkDeviceSize bufferSize = sizeof(RectDrawCommand)*MAX_RECT_COUNT;
    if(!vkCreateBufferEX(device, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,bufferSize,&out->buffer,&out->memory)) return false;
    if(vkMapMemory(device, out->memory, 0, bufferSize, 0, &out->mapped) != VK_SUCCESS) return false;
    if(!ui_create_rect_descriptor_set_and_bind_buffer(device, descriptorPool, &out->descriptorSet, out->buffer)) return false;

    return out;
}

void UIRectBufferPool_reset(UIRectBufferPool* pool){
    pool->used = 0;
}

static void ui_draw_rects(VkCommandBuffer cmd, size_t screenWidth, size_t screenHeight, VkImageView colorAttachment, VkRect2D scissor, void* mapped, VkDescriptorSet descriptorSet, RectDrawCommands* rects){
    assert(rects->count <= MAX_RECT_COUNT);

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

static RectDrawCommands tempRectDrawCommands = {0};
static UIRectBufferPool tempUIRectBufferPool = {0};
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
    UIRectBufferPool_reset(&tempUIRectBufferPool);

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
            else if(type == UI_DRAW_CMD_NONE) assert(false && "Unreachable NONE");
            else if(type == UI_DRAW_CMDS_COUNT) assert(false && "Unreachable COUNT");
            else assert(false && "Unreachable TYPE");

            index++;
            if (type == UI_DRAW_CMD_RECT && tempRectDrawCommands.count >= MAX_RECT_COUNT) break;
        }

        if(!(scissor.extent.width == 0 || scissor.extent.height == 0)) {
            //drawing
            if (type == UI_DRAW_CMD_RECT) {
                UIRectBuffer* buff = UIRectBufferPool_get_avaliable(&tempUIRectBufferPool, uiDevice, uiDescriptorPool);
                ui_draw_rects(cmd, screenWidth, screenHeight, colorAttachment, scissor, buff->mapped,buff->descriptorSet, &tempRectDrawCommands);
            }
        }

        //cleanup
        if (type == UI_DRAW_CMD_RECT) tempRectDrawCommands.count = 0;
        
        if(scissor_changed){
            scissor_changed = false;
            scissor = new_scissor;
        }
    }
}
