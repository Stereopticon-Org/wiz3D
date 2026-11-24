/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*/

// Unified compute-shader composite for every output mode on the compute rail
// (Half-SBS, Full-SBS, TAB, Full-TAB, Checker, Anaglyph, RowIL, ColumnIL).
//
// Session 1: scaffold only. CompositeComputeDispatch returns false so the
// current fragment-shader composite path in Composite.cpp stays authoritative.
// Pipeline state is lazy-init on first successful dispatch and released by
// CompositeComputeDestroyDevice at device teardown.
//
// Session 2 will fill in EnsureReady() + the dispatch body: create the
// descriptor-set layout, pipeline layout, compute pipeline from CompositeComp.inc,
// sampler, descriptor pool; then build a dispatch that binds the pipeline, writes
// the descriptor set for (left, right, dst), pushes constants (mode + stereo
// params), and dispatches ceil(w/8) x ceil(h/8) workgroups.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"

namespace wiz3d { namespace vk {

namespace {

// composite.comp, compiled by glslangValidator -V --target-env vulkan1.0 --vn kCompositeCompSpv.
#include "CompositeComp.inc"

// Keep in sync with composite.comp's push_constant block. The compute shader
// reads this verbatim so the layout cannot drift.
struct CompositePush
{
    uint32_t mode;         // OUTMODE_* id
    uint32_t swapEyes;     // 0 or 1
    uint32_t stereoOn;     // 0 or 1
    int32_t  edgeCropPx;   // Half-SBS crop, signed by convergence sign
    float    lastSep;      // sign picks the crop edge
    uint32_t pad0;
    uint32_t dstW;         // destination extent in pixels
    uint32_t dstH;
};
static_assert(sizeof(CompositePush) <= 128, "push constant block must fit the Vulkan 128-byte guaranteed minimum");

} // namespace

struct CompositeComputeState
{
    bool ready = false;
    bool failed = false;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout      pipelineLayout = VK_NULL_HANDLE;
    VkPipeline            pipeline = VK_NULL_HANDLE;
    VkShaderModule        cs = VK_NULL_HANDLE;
    VkSampler             sampler = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
};

// Public API --------------------------------------------------------------

// Session 1 stub: always returns false so CompositePresent keeps using the
// fragment-shader blit path. Session 2 replaces the body with real dispatch.
bool CompositeComputeDispatch(DeviceData* /*d*/, VkCommandBuffer /*cb*/, VkImage /*dst*/, VkImage /*left*/, VkImage /*right*/,
                              VkImageLayout /*dstLayout*/, uint32_t /*w*/, uint32_t /*h*/, uint32_t /*outputMode*/)
{
    // Keep the scaffold-stage SPIR-V referenced so the linker ships it in the
    // DLL from session 1 onwards -- saves surprise-size growth when session 2
    // first wires in the real dispatch.
    (void)kCompositeCompSpv;
    return false;
}

void CompositeComputeDestroyDevice(DeviceData* d)
{
    CompositeComputeState* c = d->compositeCompute;
    if (!c) return;
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
