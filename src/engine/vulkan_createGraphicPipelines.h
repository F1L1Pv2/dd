#ifndef TRIEX_VULKAN_CREATE_GRAPHIC_PIPELINES
#define TRIEX_VULKAN_CREATE_GRAPHIC_PIPELINES

#include <stdbool.h>
#include <vulkan/vulkan.h>

typedef struct {
    VkShaderModule vertexShader;
    VkShaderModule fragmentShader;
    size_t pushConstantsSize;
    VkPipeline* pipelineOUT;
    VkPipelineLayout* pipelineLayoutOUT;
    size_t descriptorSetLayoutCount;
    VkDescriptorSetLayout* descriptorSetLayouts;
    size_t vertexSize;
    size_t vertexInputAttributeDescriptionsCount;
    VkVertexInputAttributeDescription* vertexInputAttributeDescriptions;
    bool culling;
    bool depthTest;
    VkPrimitiveTopology topology;
    VkFormat outColorFormat;
} CreateGraphicsPipelineARGS;

bool vkCreateGraphicPipeline_opts(CreateGraphicsPipelineARGS args);

#define vkCreateGraphicPipeline(vertexShaderIn, fragmentShaderIn, pipelinePtrOut, pipelineLayoutPtrOut, outColorFormatIn, ...) vkCreateGraphicPipeline_opts((CreateGraphicsPipelineARGS){\
    .vertexShader = vertexShaderIn,\
    .fragmentShader = fragmentShaderIn,\
    .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,\
    .pipelineOUT = pipelinePtrOut,\
    .pipelineLayoutOUT = pipelineLayoutPtrOut,\
    .outColorFormat = outColorFormatIn,\
    __VA_ARGS__})

#endif