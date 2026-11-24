/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*/

// Unified compute-shader composite for every output mode on the compute rail
// (Half-SBS, Half-TAB, Anaglyph, RowIL, ColumnIL, Checker). SR Weave / Zondision
// / 3D Vision stay on their own rails; this one is for modes where wiz3D itself
// writes the final pixels.
//
// Resource layout (per device, lazy-init on first dispatch):
//   - descriptor set layout: 2x combined image sampler (L, R), 1x storage image (dst)
//   - pipeline layout + push constants (CompositePush)
//   - compute pipeline from CompositeComp.inc
//   - sampler: LINEAR filter, CLAMP_TO_EDGE
//   - descriptor pool: 32 sets, allocated once, written per dispatch (ring index)
//   - sampled scratch images (leftSrc, rightSrc) sized to the first dispatch's
//     (w, h, format). Recreated when the swapchain changes.
//   - output storage image (outImg): same extent + format, STORAGE + TRANSFER_SRC
//
// Dispatch contract: caller passes `inputLeft` and `inputRight` in TRANSFER_SRC_OPTIMAL
// and `outputImage` in TRANSFER_DST_OPTIMAL (the caller's final destination). The
// dispatch copies inputs to its internal sampled scratches, transitions them for
// sampling, dispatches, transitions outImg to TRANSFER_SRC and blits to outputImage,
// then leaves everything in the same layout it got them (inputs TRANSFER_SRC,
// output TRANSFER_DST). Caller handles the swapchain-image transition after.

#include "Layer.h"
#include "ProfileVK.h"
#include "../S3DAPI/GlobalData.h"
#include <cstring>
#include <vector>

namespace wiz3d { namespace vk {

namespace {

// composite.comp, compiled by glslangValidator -V --target-env vulkan1.0 --vn kCompositeCompSpv.
#include "CompositeComp.inc"

// Keep in sync with composite.comp's push_constant block.
struct CompositePush
{
    uint32_t mode;         // OUTMODE_* id
    uint32_t swapEyes;     // 0 or 1
    uint32_t stereoOn;     // 0 or 1
    int32_t  edgeCropPx;   // Half-SBS crop, signed by convergence sign
    float    lastSep;      // sign picks the crop edge
    uint32_t pad0;
    uint32_t dstW;
    uint32_t dstH;
};
static_assert(sizeof(CompositePush) <= 128, "push constant block must fit the Vulkan 128-byte guaranteed minimum");

constexpr uint32_t kPoolSize = 32;    // enough to cover frame-in-flight churn without recycling same frame

// Set when EdgeCropPx lives elsewhere; for now mirror the inline fragment-path heuristic. The
// fragment composite computes crop in Composite.cpp via an EdgeCropPx() helper; we pull the
// signed-separation + Vk().VKEdgeCrop values in through CompositePresent and pass as-is.

} // namespace

struct CompositeComputeState
{
    bool ready = false;
    bool failed = false;

    VkDescriptorSetLayout setLayout      = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline       = VK_NULL_HANDLE;
    VkShaderModule        cs             = VK_NULL_HANDLE;
    VkSampler             sampler        = VK_NULL_HANDLE;
    VkDescriptorPool      pool           = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> sets;
    uint32_t              nextSet = 0;

    // Scratch images sized to the current swapchain. Recreated on size/format change.
    uint32_t              scratchW = 0, scratchH = 0;
    VkFormat              scratchFormat = VK_FORMAT_UNDEFINED;
    VkImage               leftImg = VK_NULL_HANDLE, rightImg = VK_NULL_HANDLE, outImg = VK_NULL_HANDLE;
    VkImageView           leftView = VK_NULL_HANDLE, rightView = VK_NULL_HANDLE, outView = VK_NULL_HANDLE;
    VkDeviceMemory        leftMem = VK_NULL_HANDLE, rightMem = VK_NULL_HANDLE, outMem = VK_NULL_HANDLE;
};

namespace {

VkImageMemoryBarrier MakeBarrier(VkImage img, VkImageLayout from, VkImageLayout to, VkAccessFlags srcA, VkAccessFlags dstA)
{
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = srcA;
    b.dstAccessMask = dstA;
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    return b;
}

uint32_t FindMemType(DeviceData* d, uint32_t typeBits, VkMemoryPropertyFlags props)
{
    VkPhysicalDeviceMemoryProperties mp = {};
    d->inst->GetPhysicalDeviceMemoryProperties(d->physDev, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    return UINT32_MAX;
}

bool CreateScratch(DeviceData* d, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage,
                   VkImage* outImg, VkImageView* outView, VkDeviceMemory* outMem)
{
    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->CreateImage(d->device, &ci, nullptr, outImg) != VK_SUCCESS) return false;

    VkMemoryRequirements req = {};
    d->GetImageMemoryRequirements(d->device, *outImg, &req);
    const uint32_t mtype = FindMemType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (mtype == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mtype;
    if (d->AllocateMemory(d->device, &mai, nullptr, outMem) != VK_SUCCESS) return false;
    if (d->BindImageMemory(d->device, *outImg, *outMem, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = *outImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                       VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (d->CreateImageView(d->device, &vci, nullptr, outView) != VK_SUCCESS) return false;
    return true;
}

void DestroyScratchGroup(DeviceData* d, VkImage& img, VkImageView& view, VkDeviceMemory& mem)
{
    if (view) { d->DestroyImageView(d->device, view, nullptr); view = VK_NULL_HANDLE; }
    if (img)  { d->DestroyImage(d->device, img, nullptr);      img  = VK_NULL_HANDLE; }
    if (mem)  { d->FreeMemory(d->device, mem, nullptr);        mem  = VK_NULL_HANDLE; }
}

void DestroyScratches(DeviceData* d, CompositeComputeState* c)
{
    DestroyScratchGroup(d, c->leftImg,  c->leftView,  c->leftMem);
    DestroyScratchGroup(d, c->rightImg, c->rightView, c->rightMem);
    DestroyScratchGroup(d, c->outImg,   c->outView,   c->outMem);
    c->scratchW = c->scratchH = 0;
    c->scratchFormat = VK_FORMAT_UNDEFINED;
}

bool EnsureReady(DeviceData* d)
{
    CompositeComputeState* c = d->compositeCompute;
    if (!c) { d->compositeCompute = c = new CompositeComputeState(); }
    if (c->failed) return false;
    if (c->ready)  return true;

    // Descriptor set layout: 2 sampled + 1 storage, all compute visible.
    VkDescriptorSetLayoutBinding bindings[3] = {};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo slci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    slci.bindingCount = 3;
    slci.pBindings = bindings;
    if (d->CreateDescriptorSetLayout(d->device, &slci, nullptr, &c->setLayout) != VK_SUCCESS)
    { Log("CompositeCompute: CreateDescriptorSetLayout failed\n"); c->failed = true; return false; }

    // Pipeline layout: one set, push-constant block for mode + stereo params.
    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(CompositePush) };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &c->setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (d->CreatePipelineLayout(d->device, &plci, nullptr, &c->pipelineLayout) != VK_SUCCESS)
    { Log("CompositeCompute: CreatePipelineLayout failed\n"); c->failed = true; return false; }

    // Shader module from CompositeComp.inc.
    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(kCompositeCompSpv);
    smci.pCode = kCompositeCompSpv;
    if (d->CreateShaderModule(d->device, &smci, nullptr, &c->cs) != VK_SUCCESS)
    { Log("CompositeCompute: CreateShaderModule failed\n"); c->failed = true; return false; }

    // Compute pipeline.
    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = c->cs;
    cpci.stage.pName = "main";
    cpci.layout = c->pipelineLayout;
    if (d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &c->pipeline) != VK_SUCCESS)
    { Log("CompositeCompute: CreateComputePipelines failed\n"); c->failed = true; return false; }

    // Sampler: linear filtering with clamp (same as the fragment path's VK_FILTER_LINEAR blit).
    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (d->CreateSampler(d->device, &sci, nullptr, &c->sampler) != VK_SUCCESS)
    { Log("CompositeCompute: CreateSampler failed\n"); c->failed = true; return false; }

    // Descriptor pool + preallocated ring. Rewritten per dispatch; kPoolSize covers frame-in-flight churn.
    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kPoolSize * 2 };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          kPoolSize };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = kPoolSize;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    if (d->CreateDescriptorPool(d->device, &dpci, nullptr, &c->pool) != VK_SUCCESS)
    { Log("CompositeCompute: CreateDescriptorPool failed\n"); c->failed = true; return false; }

    std::vector<VkDescriptorSetLayout> layouts(kPoolSize, c->setLayout);
    c->sets.resize(kPoolSize);
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = c->pool;
    dsai.descriptorSetCount = kPoolSize;
    dsai.pSetLayouts = layouts.data();
    if (d->AllocateDescriptorSets(d->device, &dsai, c->sets.data()) != VK_SUCCESS)
    { Log("CompositeCompute: AllocateDescriptorSets failed\n"); c->failed = true; return false; }

    c->ready = true;
    Log("CompositeCompute: pipeline ready\n");
    return true;
}

bool EnsureScratches(DeviceData* d, uint32_t w, uint32_t h, VkFormat fmt)
{
    CompositeComputeState* c = d->compositeCompute;
    if (!c->ready) return false;
    if (c->scratchW == w && c->scratchH == h && c->scratchFormat == fmt && c->leftImg && c->rightImg && c->outImg)
        return true;

    DestroyScratches(d, c);

    const VkImageUsageFlags sampledUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    const VkImageUsageFlags storageUsage = VK_IMAGE_USAGE_STORAGE_BIT      | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    if (!CreateScratch(d, w, h, fmt, sampledUsage, &c->leftImg,  &c->leftView,  &c->leftMem)  ||
        !CreateScratch(d, w, h, fmt, sampledUsage, &c->rightImg, &c->rightView, &c->rightMem) ||
        !CreateScratch(d, w, h, fmt, storageUsage, &c->outImg,   &c->outView,   &c->outMem))
    {
        Log("CompositeCompute: scratch image create failed (%ux%u fmt=%d)\n", w, h, (int)fmt);
        DestroyScratches(d, c);
        return false;
    }
    c->scratchW = w;
    c->scratchH = h;
    c->scratchFormat = fmt;
    Log("CompositeCompute: scratches created %ux%u fmt=%d\n", w, h, (int)fmt);
    return true;
}

} // namespace

// Public API --------------------------------------------------------------

bool CompositeComputeDispatch(DeviceData* d, VkCommandBuffer cb, VkImage dst, VkImage left, VkImage right,
                              VkImageLayout /*dstLayout*/, uint32_t w, uint32_t h, uint32_t outputMode)
{
    if (!d || !cb || !dst || !left || !right || w == 0u || h == 0u) return false;
    if (!EnsureReady(d)) return false;

    // Need to know the destination format to size the scratches. The caller's dst has the final
    // format; we get it from the swapchain record via the caller's own bookkeeping, but since we
    // don't carry the format in our args, assume it matches the first swapchain's imageFormat.
    // CompositePresent calls us inside the swapchain-info scope, so look it up from d->swapchains.
    VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;
    for (auto& kv : d->swapchains) { fmt = kv.second.ci.imageFormat; break; }

    if (!EnsureScratches(d, w, h, fmt)) return false;

    CompositeComputeState* c = d->compositeCompute;

    // 1) Copy inputs to the sampled scratches. Caller guarantees left/right are in TRANSFER_SRC_OPTIMAL.
    //    Scratches are UNDEFINED on first use, TRANSFER_DST after previous dispatch's final transition.
    VkImageMemoryBarrier pre1[3] = {
        MakeBarrier(c->leftImg,  VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT),
        MakeBarrier(c->rightImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT),
        MakeBarrier(c->outImg,   VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,              0, VK_ACCESS_SHADER_WRITE_BIT),
    };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 3, pre1);

    VkImageCopy copy = {};
    copy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.dstSubresource = copy.srcSubresource;
    copy.extent = { w, h, 1 };
    d->CmdCopyImage(cb, left,  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c->leftImg,  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    d->CmdCopyImage(cb, right, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c->rightImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    // 2) Transition the scratches to SHADER_READ_ONLY (left, right). outImg already GENERAL.
    VkImageMemoryBarrier pre2[2] = {
        MakeBarrier(c->leftImg,  VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
        MakeBarrier(c->rightImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT),
    };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, pre2);

    // 3) Pick the next ring descriptor set and write it.
    const uint32_t slot = c->nextSet;
    c->nextSet = (c->nextSet + 1u) % kPoolSize;
    VkDescriptorSet set = c->sets[slot];

    VkDescriptorImageInfo leftInfo  = { c->sampler, c->leftView,  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo rightInfo = { c->sampler, c->rightView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo outInfo   = { VK_NULL_HANDLE, c->outView, VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet writes[3] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set;
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &leftInfo;
    writes[1] = writes[0];
    writes[1].dstBinding = 1;
    writes[1].pImageInfo = &rightInfo;
    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = set;
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &outInfo;
    d->UpdateDescriptorSets(d->device, 3, writes, 0, nullptr);

    // 4) Bind + push + dispatch.
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipeline);
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, c->pipelineLayout, 0, 1, &set, 0, nullptr);

    CompositePush push = {};
    push.mode = outputMode;
    push.swapEyes = gInfo.Input.SwapEyes ? 1u : 0u;
    push.stereoOn = gInfo.Input.StereoActive ? 1u : 0u;
    push.edgeCropPx = EdgeCropPx(d->lastSep, (int32_t)w, Vk().VKEdgeCrop);
    push.lastSep = d->lastSep;
    push.pad0 = 0;
    push.dstW = w;
    push.dstH = h;
    d->CmdPushConstants(cb, c->pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    const uint32_t gx = (w + 7u) / 8u;
    const uint32_t gy = (h + 7u) / 8u;
    d->CmdDispatch(cb, gx, gy, 1);

    // 5) Transition outImg to TRANSFER_SRC and blit to dst. Caller passed dst in TRANSFER_DST_OPTIMAL.
    VkImageMemoryBarrier post1 = MakeBarrier(c->outImg, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                             VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &post1);

    VkImageCopy blit = {};
    blit.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    blit.dstSubresource = blit.srcSubresource;
    blit.extent = { w, h, 1 };
    d->CmdCopyImage(cb, c->outImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit);

    return true;
}

void CompositeComputeDestroyDevice(DeviceData* d)
{
    CompositeComputeState* c = d->compositeCompute;
    if (!c) return;
    DestroyScratches(d, c);
    if (c->pipeline       && d->DestroyPipeline)            d->DestroyPipeline(d->device, c->pipeline, nullptr);
    if (c->pipelineLayout && d->DestroyPipelineLayout)      d->DestroyPipelineLayout(d->device, c->pipelineLayout, nullptr);
    if (c->setLayout      && d->DestroyDescriptorSetLayout) d->DestroyDescriptorSetLayout(d->device, c->setLayout, nullptr);
    if (c->pool           && d->DestroyDescriptorPool)      d->DestroyDescriptorPool(d->device, c->pool, nullptr);
    if (c->sampler        && d->DestroySampler)             d->DestroySampler(d->device, c->sampler, nullptr);
    if (c->cs             && d->DestroyShaderModule)        d->DestroyShaderModule(d->device, c->cs, nullptr);
    delete c;
    d->compositeCompute = nullptr;
}

}} // namespace wiz3d::vk
