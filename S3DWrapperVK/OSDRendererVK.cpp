/*
* Project : wiz3D
* Copyright (C) Akshay Hegde 2026
*/

// The DX9/DX11 wrappers' on-screen display (S3DWrapper10/OSDRenderer.cpp) over the presented image: the status strip,
// the FPS counter and the hotkey list, drawn by GDI into images that a quad pipeline blends over the frame.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include <algorithm>
#include <string.h>
#include <wchar.h>

#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")

namespace wiz3d { namespace vk {

namespace {

// osd.vert and osd.frag, compiled by glslangValidator -V --target-env vulkan1.0 --vn <name>.
#include "OsdVert.inc"
#include "OsdFrag.inc"
// osd_compute.comp: alpha-blend compute path for compute-only present queues (Wolf II etc.).
#include "OsdCompute.inc"

constexpr uint32_t kOsdComputeSetRing = 32;   // enough for ~10 frames in flight x 3 slots

// Keep in sync with osd_compute.comp's push_constant block.
struct OsdComputePush
{
    int32_t originX, originY;   // slot top-left on dst, in pixels
    int32_t slotW,   slotH;     // slot image extent
    int32_t dstW,    dstH;      // destination extent
    int32_t pad0,    pad1;
};
static_assert(sizeof(OsdComputePush) <= 128, "OSD push constant block must fit the Vulkan 128-byte guaranteed minimum");

enum { kStatus, kFps, kHelp, kSlotCount };

// One overlay element: the GDI bitmap its text is drawn into (BGRA, alpha kept) and the image the quad samples.
struct OsdSlot
{
    int            w = 0, h = 0;
    HDC            dc = nullptr;
    HBITMAP        dib = nullptr, prev = nullptr;
    BYTE*          bits = nullptr;
    bool           dirty = false;
    VkImage        image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView    view = VK_NULL_HANDLE;
    VkBuffer       staging = VK_NULL_HANDLE;
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    void*          mapped = nullptr;
    VkDescriptorSet set = VK_NULL_HANDLE;
};

struct OsdPass { VkRenderPass pass = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE; };
struct OsdTarget { VkImageView view = VK_NULL_HANDLE; VkFramebuffer fb = VK_NULL_HANDLE; VkFormat format = VK_FORMAT_UNDEFINED; uint32_t w = 0, h = 0; };

} // namespace

struct OsdState
{
    bool ready = false, failed = false;
    OsdSlot slots[kSlotCount];
    HFONT fontStatus = nullptr, fontFps = nullptr, fontHelp = nullptr, fontHelpBody = nullptr;
    VkSampler             sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkDescriptorPool      pool = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkShaderModule        vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    std::unordered_map<int, OsdPass> passes;          // by the presented image's format; a null pass failed
    std::unordered_map<VkImage, OsdTarget> targets;   // by presented image
    // Compute OSD path (compute-only queue families). Parallel to the graphics pipeline above;
    // reuses `sampler` and the per-slot `OsdSlot` images. Lazy-init on first OsdDrawCompute call.
    bool                  cReady = false, cFailed = false;
    VkShaderModule        cs = VK_NULL_HANDLE;
    VkDescriptorSetLayout cSetLayout = VK_NULL_HANDLE;
    VkPipelineLayout      cPipelineLayout = VK_NULL_HANDLE;
    VkPipeline            cPipeline = VK_NULL_HANDLE;
    VkDescriptorPool      cPool = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> cSets;
    uint32_t              cNextSet = 0;
    // Scratch image that compute writes the alpha-blended output to, then copies to the swapchain
    // image (avoids needing STORAGE_BIT on the swapchain image, which some present engines refuse).
    uint32_t              scratchW = 0, scratchH = 0;
    VkFormat              scratchFormat = VK_FORMAT_UNDEFINED;
    VkImage               scratchImg = VK_NULL_HANDLE;
    VkImageView           scratchView = VK_NULL_HANDLE;
    VkDeviceMemory        scratchMem = VK_NULL_HANDLE;
    // FPS: presents counted over 250 ms windows, smoothed.
    ULONGLONG lastFpsSample = 0;
    unsigned  framesSinceSample = 0;
    float     fpsEma = 0.f;
    // The status strip shows for two seconds after a stereo setting changes (ShowOSD 1).
    ULONGLONG statusUntil = 0;
    unsigned  lastPreset = (unsigned)-1;
    float     lastStereoBase = -1.f, lastOneDivZps = -1.f;
    bool      lastSwapEyes = false;
    // What each slot last drew, so unchanged text is not drawn again.
    wchar_t lastStatus[128] = {};
    wchar_t lastFps[32] = {};
    bool    helpDrawn = false;
};

namespace {

HFONT MakeFont(int ptSize, int weight)
{
    // Grey antialiasing: the outline pass reads one channel as coverage, which ClearType's subpixels would break.
    return CreateFontW(-MulDiv(ptSize, 96, 72), 0, 0, 0, weight, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_TT_PRECIS,
                       CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, VARIABLE_PITCH | FF_SWISS, L"Arial");
}

void SlotClear(OsdSlot& s)
{
    if (!s.bits) return;
    memset(s.bits, 0, (size_t)s.w * s.h * 4);
    s.dirty = true;
}

// White text with a two-pixel black outline and no panel: GDI draws white on black, that grey level is the glyph's
// coverage, and each pixel becomes the fill at its own coverage or black at the highest coverage within two pixels.
void SlotDrawText(OsdSlot& s, const wchar_t* text, const RECT& r, HFONT font, COLORREF rgb, UINT flags)
{
    if (!s.dc || !s.bits || !font || !text) return;
    const int x0 = (std::max)(0L, (long)r.left), x1 = (std::min)((long)s.w, (long)r.right);
    const int y0 = (std::max)(0L, (long)r.top), y1 = (std::min)((long)s.h, (long)r.bottom);
    const int rw = x1 - x0, rh = y1 - y0;
    if (rw <= 0 || rh <= 0) return;
    HFONT prevFont = (HFONT)SelectObject(s.dc, font);
    for (int y = y0; y < y1; ++y) memset(s.bits + ((size_t)y * s.w + x0) * 4, 0, (size_t)rw * 4);
    SetTextColor(s.dc, RGB(255, 255, 255));
    RECT rc = r;
    DrawTextW(s.dc, text, -1, &rc, flags);
    std::vector<BYTE> mask((size_t)rw * rh, 0);
    for (int y = 0; y < rh; ++y)
    {
        const BYTE* row = s.bits + ((size_t)(y + y0) * s.w + x0) * 4;
        for (int x = 0; x < rw; ++x) mask[(size_t)y * rw + x] = row[x * 4 + 2];
    }
    const BYTE fr = GetRValue(rgb), fg = GetGValue(rgb), fb = GetBValue(rgb);
    const int kOutline = 2;
    for (int y = 0; y < rh; ++y)
    {
        BYTE* row = s.bits + ((size_t)(y + y0) * s.w + x0) * 4;
        for (int x = 0; x < rw; ++x)
        {
            BYTE outline = 0;
            for (int dy = -kOutline; dy <= kOutline && outline < 255; ++dy)
            {
                const int ny = y + dy;
                if (ny < 0 || ny >= rh) continue;
                for (int dx = -kOutline; dx <= kOutline && outline < 255; ++dx)
                {
                    const int nx = x + dx;
                    if (nx < 0 || nx >= rw || dx * dx + dy * dy > kOutline * kOutline) continue;
                    outline = (std::max)(outline, mask[(size_t)ny * rw + nx]);
                }
            }
            const BYTE fill = mask[(size_t)y * rw + x];
            BYTE* p = row + x * 4;
            if (fill) { p[0] = fb; p[1] = fg; p[2] = fr; p[3] = fill; }
            else { p[0] = p[1] = p[2] = 0; p[3] = outline; }
        }
    }
    SelectObject(s.dc, prevFont);
    s.dirty = true;
}

bool SlotCreate(DeviceData* d, OsdState* o, OsdSlot& s, int width, int height)
{
    s.w = width; s.h = height;
    HDC screen = GetDC(nullptr);
    s.dc = CreateCompatibleDC(screen);
    ReleaseDC(nullptr, screen);
    if (!s.dc) return false;
    // Top-down, 32 bits: the bitmap's rows and bytes are the image's.
    BITMAPINFO bmi = {};
    bmi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
    bmi.bmiHeader.biWidth = width;
    bmi.bmiHeader.biHeight = -height;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biCompression = BI_RGB;
    void* bits = nullptr;
    s.dib = CreateDIBSection(s.dc, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
    if (!s.dib || !bits) return false;
    s.bits = (BYTE*)bits;
    s.prev = (HBITMAP)SelectObject(s.dc, s.dib);
    SetBkMode(s.dc, TRANSPARENT);

    VkImageCreateInfo ici = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = VK_FORMAT_B8G8R8A8_UNORM;
    ici.extent = { (uint32_t)width, (uint32_t)height, 1 };
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->CreateImage(d->device, &ici, nullptr, &s.image) != VK_SUCCESS) return false;
    VkMemoryRequirements req;
    d->GetImageMemoryRequirements(d->device, s.image, &req);
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, &s.memory) != VK_SUCCESS) return false;
    if (d->BindImageMemory(d->device, s.image, s.memory, 0) != VK_SUCCESS) return false;
    VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = s.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = ici.format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (d->CreateImageView(d->device, &vci, nullptr, &s.view) != VK_SUCCESS) return false;

    VkBufferCreateInfo bci = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bci.size = (VkDeviceSize)width * height * 4;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (d->CreateBuffer(d->device, &bci, nullptr, &s.staging) != VK_SUCCESS) return false;
    d->GetBufferMemoryRequirements(d->device, s.staging, &req);
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = FindMemoryType(d, req.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, UINT32_MAX);
    if (mai.memoryTypeIndex == UINT32_MAX || d->AllocateMemory(d->device, &mai, nullptr, &s.stagingMemory) != VK_SUCCESS) return false;
    if (d->BindBufferMemory(d->device, s.staging, s.stagingMemory, 0) != VK_SUCCESS) return false;
    if (d->MapMemory(d->device, s.stagingMemory, 0, VK_WHOLE_SIZE, 0, &s.mapped) != VK_SUCCESS) return false;

    VkDescriptorSetAllocateInfo dai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dai.descriptorPool = o->pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &o->setLayout;
    if (d->AllocateDescriptorSets(d->device, &dai, &s.set) != VK_SUCCESS) return false;
    const VkDescriptorImageInfo dii = { o->sampler, s.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkWriteDescriptorSet w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
    w.dstSet = s.set;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &dii;
    d->UpdateDescriptorSets(d->device, 1, &w, 0, nullptr);
    SlotClear(s);
    return true;
}

void SlotDestroy(DeviceData* d, OsdSlot& s)
{
    if (s.view) d->DestroyImageView(d->device, s.view, nullptr);
    if (s.image) d->DestroyImage(d->device, s.image, nullptr);
    if (s.memory) d->FreeMemory(d->device, s.memory, nullptr);
    if (s.mapped) d->UnmapMemory(d->device, s.stagingMemory);
    if (s.staging) d->DestroyBuffer(d->device, s.staging, nullptr);
    if (s.stagingMemory) d->FreeMemory(d->device, s.stagingMemory, nullptr);
    if (s.dc && s.prev) SelectObject(s.dc, s.prev);
    if (s.dib) DeleteObject(s.dib);
    if (s.dc) DeleteDC(s.dc);
    s = OsdSlot();
}

// The slot's bitmap into its image, whole: the image's old contents and layout do not matter.
void SlotUpload(DeviceData* d, VkCommandBuffer cb, OsdSlot& s)
{
    if (!s.dirty) return;
    s.dirty = false;
    memcpy(s.mapped, s.bits, (size_t)s.w * s.h * 4);
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = s.image;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkBufferImageCopy copy = {};
    copy.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copy.imageExtent = { (uint32_t)s.w, (uint32_t)s.h, 1 };
    d->CmdCopyBufferToImage(cb, s.staging, s.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    // Covers both the fragment OSD path and the compute OSD path (OsdDrawCompute).
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &b);
}

// Fonts, the three slots at DX11's sizes, and everything the pipelines share.
bool Init(DeviceData* d, OsdState* o)
{
    o->fontStatus = MakeFont(20, FW_SEMIBOLD);
    o->fontFps = MakeFont(16, FW_SEMIBOLD);
    o->fontHelp = MakeFont(22, FW_BOLD);
    o->fontHelpBody = MakeFont(15, FW_NORMAL);
    if (!o->fontStatus || !o->fontFps || !o->fontHelp || !o->fontHelpBody) return false;

    VkSamplerCreateInfo sci = { VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (!d->CreateSampler || d->CreateSampler(d->device, &sci, nullptr, &o->sampler) != VK_SUCCESS) return false;
    VkDescriptorSetLayoutBinding bind = {};
    bind.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bind.descriptorCount = 1;
    bind.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo lci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    lci.bindingCount = 1;
    lci.pBindings = &bind;
    if (d->CreateDescriptorSetLayout(d->device, &lci, nullptr, &o->setLayout) != VK_SUCCESS) return false;
    const VkDescriptorPoolSize size = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kSlotCount };
    VkDescriptorPoolCreateInfo pci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    pci.maxSets = kSlotCount;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &size;
    if (d->CreateDescriptorPool(d->device, &pci, nullptr, &o->pool) != VK_SUCCESS) return false;
    const VkPushConstantRange range = { VK_SHADER_STAGE_VERTEX_BIT, 0, 16 };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &o->setLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &range;
    if (d->CreatePipelineLayout(d->device, &plci, nullptr, &o->layout) != VK_SUCCESS) return false;
    VkShaderModuleCreateInfo mci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    mci.codeSize = sizeof(kOsdVertSpv);
    mci.pCode = kOsdVertSpv;
    if (d->CreateShaderModule(d->device, &mci, nullptr, &o->vs) != VK_SUCCESS) return false;
    mci.codeSize = sizeof(kOsdFragSpv);
    mci.pCode = kOsdFragSpv;
    if (d->CreateShaderModule(d->device, &mci, nullptr, &o->fs) != VK_SUCCESS) return false;

    static const int kSize[kSlotCount][2] = { { 1100, 48 }, { 180, 36 }, { 720, 520 } };
    for (int i = 0; i < kSlotCount; ++i)
        if (!SlotCreate(d, o, o->slots[i], kSize[i][0], kSize[i][1])) return false;
    return true;
}

// One render pass and pipeline per presented format: the frame is kept, the quad blends over it by its alpha.
OsdPass* EnsurePass(DeviceData* d, OsdState* o, VkFormat format)
{
    auto it = o->passes.find((int)format);
    if (it != o->passes.end()) return it->second.pass ? &it->second : nullptr;
    OsdPass& p = o->passes[(int)format];
    VkAttachmentDescription att = {};
    att.format = format;
    att.samples = VK_SAMPLE_COUNT_1_BIT;
    att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    att.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    att.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    att.initialLayout = att.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    const VkAttachmentReference ref = { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkSubpassDescription sub = {};
    sub.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    sub.colorAttachmentCount = 1;
    sub.pColorAttachments = &ref;
    VkRenderPassCreateInfo rci = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
    rci.attachmentCount = 1;
    rci.pAttachments = &att;
    rci.subpassCount = 1;
    rci.pSubpasses = &sub;
    if (d->CreateRenderPass(d->device, &rci, nullptr, &p.pass) != VK_SUCCESS) { p.pass = VK_NULL_HANDLE; }

    VkPipelineShaderStageCreateInfo stages[2] = {};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = o->vs;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = o->fs;
    stages[0].pName = stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vi = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
    VkPipelineInputAssemblyStateCreateInfo ia = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo vp = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    // DX11's blend state: colour over by source alpha, alpha kept as one minus it.
    VkPipelineColorBlendAttachmentState ba = {};
    ba.blendEnable = VK_TRUE;
    ba.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    ba.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.colorBlendOp = VK_BLEND_OP_ADD;
    ba.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    ba.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    ba.alphaBlendOp = VK_BLEND_OP_ADD;
    ba.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo cb = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
    cb.attachmentCount = 1;
    cb.pAttachments = &ba;
    const VkDynamicState dyn[2] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    VkPipelineDynamicStateCreateInfo ds = { VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
    ds.dynamicStateCount = 2;
    ds.pDynamicStates = dyn;
    VkGraphicsPipelineCreateInfo gci = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
    gci.stageCount = 2;
    gci.pStages = stages;
    gci.pVertexInputState = &vi;
    gci.pInputAssemblyState = &ia;
    gci.pViewportState = &vp;
    gci.pRasterizationState = &rs;
    gci.pMultisampleState = &ms;
    gci.pColorBlendState = &cb;
    gci.pDynamicState = &ds;
    gci.layout = o->layout;
    gci.renderPass = p.pass;
    if (p.pass && d->CreateGraphicsPipelines(d->device, VK_NULL_HANDLE, 1, &gci, nullptr, &p.pipeline) != VK_SUCCESS)
    {
        d->DestroyRenderPass(d->device, p.pass, nullptr);
        p.pass = VK_NULL_HANDLE;
    }
    if (!p.pass) { Log("OSD: no pipeline for format %d: nothing is drawn over images of it\n", (int)format); return nullptr; }
    return &p;
}

void TargetDestroy(DeviceData* d, OsdTarget& t)
{
    if (t.fb) d->DestroyFramebuffer(d->device, t.fb, nullptr);
    if (t.view) d->DestroyImageView(d->device, t.view, nullptr);
    t = OsdTarget();
}

// The presented image as a framebuffer, made once per image and again when its shape changed under the same handle.
OsdTarget* EnsureTarget(DeviceData* d, OsdState* o, const OsdPass& pass, VkImage image, VkFormat format, uint32_t w, uint32_t h)
{
    OsdTarget& t = o->targets[image];
    if (t.fb && t.format == format && t.w == w && t.h == h) return &t;
    TargetDestroy(d, t);
    VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (d->CreateImageView(d->device, &vci, nullptr, &t.view) != VK_SUCCESS) { t = OsdTarget(); return nullptr; }
    VkFramebufferCreateInfo fci = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
    fci.renderPass = pass.pass;
    fci.attachmentCount = 1;
    fci.pAttachments = &t.view;
    fci.width = w;
    fci.height = h;
    fci.layers = 1;
    if (d->CreateFramebuffer(d->device, &fci, nullptr, &t.fb) != VK_SUCCESS) { TargetDestroy(d, t); return nullptr; }
    t.format = format; t.w = w; t.h = h;
    return &t;
}

// ShowOSD: 2 shows the strip always, 1 for two seconds after a setting changed, 0 never.
bool WantStatus(const OsdState* o)
{
    if (gInfo.ShowOSD >= 2) return true;
    if (gInfo.ShowOSD == 0) return false;
    return GetTickCount64() < o->statusUntil;
}

// DX11's strip: "Conv" is the convergence shift, "Screen Plane" the zero-parallax depth 1/One_div_ZPS.
void BuildStatusText(wchar_t* out, size_t cap)
{
    const CameraPreset* p = gInfo.Input.GetActivePreset();
    const float sepPercent = p ? p->StereoBase / 0.008f : 0.f;
    const float screenPlane = (p && p->One_div_ZPS != 0.f) ? 1.f / p->One_div_ZPS : 0.f;
    _snwprintf_s(out, cap, _TRUNCATE, L"wiz3D   |   Preset %d   |   Sep %.0f%%   |   Conv %+.2f   |   Screen Plane %.2f%s%s",
                 (int)gInfo.Input.ActivePreset + 1, sepPercent, gInfo.Input.ConvergenceShift, screenPlane,
                 (p && p->AutoFocusEnable) ? L"   |   AutoFocus" : L"", gInfo.Input.SwapEyes ? L"   |   Eyes Swapped" : L"");
}

// The texts whose content changed since they were last drawn.
void UpdateSlots(OsdState* o, bool wantStatus, bool wantFps, bool wantHelp)
{
    const UINT kLine = DT_SINGLELINE | DT_NOPREFIX;
    if (wantStatus)
    {
        wchar_t text[128];
        BuildStatusText(text, _countof(text));
        if (wcscmp(text, o->lastStatus) != 0)
        {
            OsdSlot& s = o->slots[kStatus];
            SlotClear(s);
            const RECT r = { 12, 4, s.w - 12, s.h - 4 };
            SlotDrawText(s, text, r, o->fontStatus, RGB(255, 255, 255), DT_LEFT | DT_VCENTER | kLine);
            wcscpy_s(o->lastStatus, text);
        }
    }
    if (wantFps)
    {
        wchar_t text[32];
        _snwprintf_s(text, _countof(text), _TRUNCATE, L"%.0f fps", o->fpsEma);
        if (wcscmp(text, o->lastFps) != 0)
        {
            OsdSlot& s = o->slots[kFps];
            SlotClear(s);
            const RECT r = { 8, 2, s.w - 8, s.h - 2 };
            SlotDrawText(s, text, r, o->fontFps, RGB(255, 255, 255), DT_RIGHT | DT_VCENTER | kLine);
            wcscpy_s(o->lastFps, text);
        }
    }
    if (wantHelp && !o->helpDrawn)
    {
        // DX11's list, less the hotkeys this wrapper has nothing behind.
        static const wchar_t* const kLines[] = {
            L"Numpad *",             L"Toggle stereo on / off",
            L"Numpad +  /  -",       L"Separation (depth strength)",
            L"Shift + Numpad +/-",   L"Screen Plane (where zero-parallax depth sits)",
            L"Numpad 7 / 8 / 9",     L"Preset Low / Default / Strong",
            L"Ctrl + F12",           L"Stereo screenshot",
            L"Shift + F1",           L"Toggle this hotkey list",
            L"Ctrl + F1",            L"Toggle OSD strip + FPS",
            L"Ctrl + F5",            L"Reset preset values",
        };
        OsdSlot& s = o->slots[kHelp];
        SlotClear(s);
        const RECT title = { 20, 20, s.w - 20, 60 };
        SlotDrawText(s, L"wiz3D Hotkeys", title, o->fontHelp, RGB(255, 255, 255), DT_LEFT | kLine);
        const int rowH = 32, keyX = 32, valueX = 240;
        int row = 72;
        for (int i = 0; i < (int)_countof(kLines); i += 2, row += rowH)
        {
            const RECT k = { keyX, row, valueX - 8, row + rowH };
            const RECT v = { valueX, row, s.w - 20, row + rowH };
            SlotDrawText(s, kLines[i], k, o->fontHelpBody, RGB(180, 220, 255), DT_LEFT | kLine);
            SlotDrawText(s, kLines[i + 1], v, o->fontHelpBody, RGB(230, 230, 230), DT_LEFT | kLine);
        }
        o->helpDrawn = true;
    }
    if (!wantHelp) o->helpDrawn = false;
}

void DrawQuad(DeviceData* d, OsdState* o, VkCommandBuffer cb, const OsdSlot& s, float x, float y, float w, float h, uint32_t bbW, uint32_t bbH)
{
    const float rect[4] = { x / bbW * 2.f - 1.f, y / bbH * 2.f - 1.f, (x + w) / bbW * 2.f - 1.f, (y + h) / bbH * 2.f - 1.f };
    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, o->layout, 0, 1, &s.set, 0, nullptr);
    d->CmdPushConstants(cb, o->layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(rect), rect);
    d->CmdDraw(cb, 4, 1, 0, 0);
}

// Compute OSD: alpha-blend text slots over the presented image on queues without graphics.
// Init() runs first (shared sampler + slots). This is strictly additive -- the fragment path is unchanged.
bool InitCompute(DeviceData* d, OsdState* o)
{
    if (o->cReady)  return true;
    if (o->cFailed) return false;
    if (!o->ready)  return false;    // Init() must have run (sampler, slots)

    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0] = { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
    VkDescriptorSetLayoutCreateInfo slci = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    slci.bindingCount = 2;
    slci.pBindings = bindings;
    if (d->CreateDescriptorSetLayout(d->device, &slci, nullptr, &o->cSetLayout) != VK_SUCCESS) return false;

    VkPushConstantRange pcr = { VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(OsdComputePush) };
    VkPipelineLayoutCreateInfo plci = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &o->cSetLayout;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (d->CreatePipelineLayout(d->device, &plci, nullptr, &o->cPipelineLayout) != VK_SUCCESS) return false;

    VkShaderModuleCreateInfo smci = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = sizeof(kOsdComputeSpv);
    smci.pCode = kOsdComputeSpv;
    if (d->CreateShaderModule(d->device, &smci, nullptr, &o->cs) != VK_SUCCESS) return false;

    VkComputePipelineCreateInfo cpci = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = o->cs;
    cpci.stage.pName = "main";
    cpci.layout = o->cPipelineLayout;
    if (d->CreateComputePipelines(d->device, VK_NULL_HANDLE, 1, &cpci, nullptr, &o->cPipeline) != VK_SUCCESS) return false;

    VkDescriptorPoolSize poolSizes[2] = {};
    poolSizes[0] = { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kOsdComputeSetRing };
    poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,          kOsdComputeSetRing };
    VkDescriptorPoolCreateInfo dpci = { VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = kOsdComputeSetRing;
    dpci.poolSizeCount = 2;
    dpci.pPoolSizes = poolSizes;
    if (d->CreateDescriptorPool(d->device, &dpci, nullptr, &o->cPool) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayout> layouts(kOsdComputeSetRing, o->cSetLayout);
    o->cSets.resize(kOsdComputeSetRing);
    VkDescriptorSetAllocateInfo dsai = { VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = o->cPool;
    dsai.descriptorSetCount = kOsdComputeSetRing;
    dsai.pSetLayouts = layouts.data();
    if (d->AllocateDescriptorSets(d->device, &dsai, o->cSets.data()) != VK_SUCCESS) return false;

    o->cReady = true;
    return true;
}

bool EnsureComputeScratch(DeviceData* d, OsdState* o, uint32_t w, uint32_t h, VkFormat fmt)
{
    if (o->scratchW == w && o->scratchH == h && o->scratchFormat == fmt && o->scratchImg) return true;

    if (o->scratchView) { d->DestroyImageView(d->device, o->scratchView, nullptr); o->scratchView = VK_NULL_HANDLE; }
    if (o->scratchImg)  { d->DestroyImage(d->device, o->scratchImg, nullptr);       o->scratchImg  = VK_NULL_HANDLE; }
    if (o->scratchMem)  { d->FreeMemory(d->device, o->scratchMem, nullptr);         o->scratchMem  = VK_NULL_HANDLE; }

    VkImageCreateInfo ci = { VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = { w, h, 1 };
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (d->CreateImage(d->device, &ci, nullptr, &o->scratchImg) != VK_SUCCESS) return false;

    VkMemoryRequirements req = {};
    d->GetImageMemoryRequirements(d->device, o->scratchImg, &req);
    VkPhysicalDeviceMemoryProperties mp = {};
    d->inst->GetPhysicalDeviceMemoryProperties(d->physDev, &mp);
    uint32_t mtype = UINT32_MAX;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((req.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) { mtype = i; break; }
    if (mtype == UINT32_MAX) return false;
    VkMemoryAllocateInfo mai = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = mtype;
    if (d->AllocateMemory(d->device, &mai, nullptr, &o->scratchMem) != VK_SUCCESS) return false;
    if (d->BindImageMemory(d->device, o->scratchImg, o->scratchMem, 0) != VK_SUCCESS) return false;

    VkImageViewCreateInfo vci = { VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    vci.image = o->scratchImg;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = fmt;
    vci.components = { VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY };
    vci.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    if (d->CreateImageView(d->device, &vci, nullptr, &o->scratchView) != VK_SUCCESS) return false;

    o->scratchW = w; o->scratchH = h; o->scratchFormat = fmt;
    return true;
}

void DispatchSlotCompute(DeviceData* d, OsdState* o, VkCommandBuffer cb, const OsdSlot& s,
                         int32_t x, int32_t y, int32_t sw, int32_t sh, uint32_t dstW, uint32_t dstH)
{
    const uint32_t slot = o->cNextSet;
    o->cNextSet = (o->cNextSet + 1u) % kOsdComputeSetRing;
    VkDescriptorSet set = o->cSets[slot];

    VkDescriptorImageInfo slotInfo = { o->sampler, s.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL };
    VkDescriptorImageInfo dstInfo  = { VK_NULL_HANDLE, o->scratchView, VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet writes[2] = {};
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    writes[0].pImageInfo = &slotInfo;
    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &dstInfo;
    d->UpdateDescriptorSets(d->device, 2, writes, 0, nullptr);

    d->CmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, o->cPipelineLayout, 0, 1, &set, 0, nullptr);

    OsdComputePush push = {};
    push.originX = x; push.originY = y;
    push.slotW   = sw; push.slotH  = sh;
    push.dstW    = (int32_t)dstW; push.dstH = (int32_t)dstH;
    d->CmdPushConstants(cb, o->cPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    const uint32_t gx = ((uint32_t)sw + 7u) / 8u;
    const uint32_t gy = ((uint32_t)sh + 7u) / 8u;
    d->CmdDispatch(cb, gx, gy, 1);

    // RAW on scratchImg between dispatches (each slot reads + writes).
    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = o->scratchImg;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          0, 0, nullptr, 0, nullptr, 1, &b);
}

} // namespace

// Once per present: the FPS average, and the two-second status strip when a stereo setting changed since the last.
void OsdTick(DeviceData* d)
{
    if (!d->osd) d->osd = new OsdState();
    OsdState* o = d->osd;
    const ULONGLONG now = GetTickCount64();
    ++o->framesSinceSample;
    if (!o->lastFpsSample) o->lastFpsSample = now;
    if (now - o->lastFpsSample >= 250)
    {
        const float fps = (float)(o->framesSinceSample * 1000.0 / (double)(now - o->lastFpsSample));
        o->fpsEma = o->fpsEma == 0.f ? fps : 0.6f * fps + 0.4f * o->fpsEma;
        o->framesSinceSample = 0;
        o->lastFpsSample = now;
    }
    const CameraPreset* p = gInfo.Input.GetActivePreset();
    const float stereoBase = p ? p->StereoBase : 0.f, oneDivZps = p ? p->One_div_ZPS : 0.f;
    const bool swapEyes = gInfo.Input.SwapEyes != 0;
    const unsigned preset = gInfo.Input.ActivePreset;
    const bool first = o->lastPreset == (unsigned)-1;
    if (!first && (preset != o->lastPreset || stereoBase != o->lastStereoBase || oneDivZps != o->lastOneDivZps || swapEyes != o->lastSwapEyes))
        o->statusUntil = now + 2000;
    o->lastPreset = preset;
    o->lastStereoBase = stereoBase;
    o->lastOneDivZps = oneDivZps;
    o->lastSwapEyes = swapEyes;
}

bool OsdWanted(DeviceData* d)
{
    const OsdState* o = d->osd;
    return o && !o->failed && (WantStatus(o) || gInfo.Input.ShowFPS || gInfo.Input.ShowHotKeyOSD);
}

// Lock held. dst is the presented image, w x h, in TRANSFER_DST_OPTIMAL before and after.
void OsdDraw(DeviceData* d, VkCommandBuffer cb, VkImage dst, VkFormat format, uint32_t w, uint32_t h)
{
    OsdState* o = d->osd;
    if (!o || o->failed || !w || !h) return;
    const bool wantStatus = WantStatus(o), wantFps = gInfo.Input.ShowFPS != 0, wantHelp = gInfo.Input.ShowHotKeyOSD != 0;
    if (!wantStatus && !wantFps && !wantHelp) return;
    if (!o->ready)
    {
        if (!Init(d, o)) { o->failed = true; Log("OSD: setup failed: no on-screen display in this session\n"); return; }
        o->ready = true;
        Log("OSD: status strip, FPS counter and hotkey list over the presented image\n");
    }
    OsdPass* pass = EnsurePass(d, o, format);
    OsdTarget* target = pass ? EnsureTarget(d, o, *pass, dst, format, w, h) : nullptr;
    if (!target) return;
    UpdateSlots(o, wantStatus, wantFps, wantHelp);
    if (wantStatus) SlotUpload(d, cb, o->slots[kStatus]);
    if (wantFps) SlotUpload(d, cb, o->slots[kFps]);
    if (wantHelp) SlotUpload(d, cb, o->slots[kHelp]);

    VkImageMemoryBarrier b = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = dst;
    b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    VkRenderPassBeginInfo rbi = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
    rbi.renderPass = pass->pass;
    rbi.framebuffer = target->fb;
    rbi.renderArea = { { 0, 0 }, { w, h } };
    d->CmdBeginRenderPass(cb, &rbi, VK_SUBPASS_CONTENTS_INLINE);
    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->pipeline);
    const VkViewport vp = { 0.f, 0.f, (float)w, (float)h, 0.f, 1.f };
    d->CmdSetViewport(cb, 0, 1, &vp);
    d->CmdSetScissor(cb, 0, 1, &rbi.renderArea);

    // DX11's placement: text clears the comfort bars at the side edges and a 16:9 picture's letterbox at the top.
    float barX = 0.f, letterboxY = 0.f;
    if (gInfo.Input.StereoActive && gInfo.FrustumAdjustMode)
        barX = gInfo.BlackAreaWidth * (float)w * (gInfo.SeparationMode == 1 ? 2.f : 1.f);
    if (gInfo.FixNone16x9TextPosition) letterboxY = (std::max)(0.f, ((float)h - (float)w / (16.f / 9.f)) * 0.5f);
    if (wantHelp)
    {
        // Centred at its own size, smaller only where the image is.
        const OsdSlot& s = o->slots[kHelp];
        const float dw = (float)(std::min)(w > 40 ? w - 40 : (uint32_t)s.w, (uint32_t)s.w);
        const float dh = (float)(std::min)(h > 40 ? h - 40 : (uint32_t)s.h, (uint32_t)s.h);
        DrawQuad(d, o, cb, s, ((float)w - dw) * 0.5f, ((float)h - dh) * 0.5f, dw, dh, w, h);
    }
    if (wantStatus)
    {
        const OsdSlot& s = o->slots[kStatus];
        DrawQuad(d, o, cb, s, 20.f + barX, 20.f + letterboxY, (float)s.w, (float)s.h, w, h);
    }
    if (wantFps)
    {
        const OsdSlot& s = o->slots[kFps];
        DrawQuad(d, o, cb, s, (float)w - 20.f - barX - (float)s.w, 20.f + letterboxY, (float)s.w, (float)s.h, w, h);
    }
    d->CmdEndRenderPass(cb);
    b.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// Lock held. dst is the presented image, w x h, in TRANSFER_DST_OPTIMAL before and after.
// Used on queue families without graphics (compute-only present on Wolfenstein II etc.).
// Mirrors OsdDraw's slot placement but goes through a compute pipeline + scratch image to
// avoid needing a render pass (impossible on compute-only queues).
void OsdDrawCompute(DeviceData* d, VkCommandBuffer cb, VkImage dst, VkFormat format, uint32_t w, uint32_t h)
{
    OsdState* o = d->osd;
    if (!o || o->failed || !w || !h) return;
    const bool wantStatus = WantStatus(o), wantFps = gInfo.Input.ShowFPS != 0, wantHelp = gInfo.Input.ShowHotKeyOSD != 0;
    if (!wantStatus && !wantFps && !wantHelp) return;
    if (!o->ready)
    {
        if (!Init(d, o)) { o->failed = true; Log("OSD: setup failed: no on-screen display in this session\n"); return; }
        o->ready = true;
        Log("OSD: status strip, FPS counter and hotkey list over the presented image\n");
    }
    if (!o->cReady && !o->cFailed)
    {
        if (!InitCompute(d, o)) { o->cFailed = true; Log("OSD compute: setup failed: no OSD on compute-only queues\n"); return; }
        Log("OSD compute: ready for compute-only present queues\n");
    }
    if (o->cFailed) return;
    if (!EnsureComputeScratch(d, o, w, h, format)) return;

    UpdateSlots(o, wantStatus, wantFps, wantHelp);
    if (wantStatus) SlotUpload(d, cb, o->slots[kStatus]);
    if (wantFps)    SlotUpload(d, cb, o->slots[kFps]);
    if (wantHelp)   SlotUpload(d, cb, o->slots[kHelp]);

    // dst TRANSFER_DST -> TRANSFER_SRC, scratch UNDEFINED -> TRANSFER_DST
    VkImageMemoryBarrier pre1[2] = {};
    pre1[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    pre1[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre1[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    pre1[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre1[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    pre1[0].srcQueueFamilyIndex = pre1[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre1[0].image = dst;
    pre1[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    pre1[1] = pre1[0];
    pre1[1].srcAccessMask = 0;
    pre1[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre1[1].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    pre1[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre1[1].image = o->scratchImg;
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 2, pre1);

    VkImageCopy copyIn = {};
    copyIn.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyIn.dstSubresource = copyIn.srcSubresource;
    copyIn.extent = { w, h, 1 };
    d->CmdCopyImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, o->scratchImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyIn);

    // scratch TRANSFER_DST -> GENERAL for compute read+write
    VkImageMemoryBarrier pre2 = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    pre2.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    pre2.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    pre2.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    pre2.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    pre2.srcQueueFamilyIndex = pre2.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre2.image = o->scratchImg;
    pre2.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    d->CmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &pre2);

    d->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, o->cPipeline);

    // Same placement rules as the fragment OsdDraw.
    float barX = 0.f, letterboxY = 0.f;
    if (gInfo.Input.StereoActive && gInfo.FrustumAdjustMode)
        barX = gInfo.BlackAreaWidth * (float)w * (gInfo.SeparationMode == 1 ? 2.f : 1.f);
    if (gInfo.FixNone16x9TextPosition)
        letterboxY = (std::max)(0.f, ((float)h - (float)w / (16.f / 9.f)) * 0.5f);

    if (wantHelp)
    {
        const OsdSlot& s = o->slots[kHelp];
        const int32_t dw = (int32_t)(std::min)(w > 40u ? w - 40u : (uint32_t)s.w, (uint32_t)s.w);
        const int32_t dh = (int32_t)(std::min)(h > 40u ? h - 40u : (uint32_t)s.h, (uint32_t)s.h);
        DispatchSlotCompute(d, o, cb, s, ((int32_t)w - dw) / 2, ((int32_t)h - dh) / 2, dw, dh, w, h);
    }
    if (wantStatus)
    {
        const OsdSlot& s = o->slots[kStatus];
        DispatchSlotCompute(d, o, cb, s, 20 + (int32_t)barX, 20 + (int32_t)letterboxY, s.w, s.h, w, h);
    }
    if (wantFps)
    {
        const OsdSlot& s = o->slots[kFps];
        DispatchSlotCompute(d, o, cb, s, (int32_t)w - 20 - (int32_t)barX - s.w, 20 + (int32_t)letterboxY, s.w, s.h, w, h);
    }

    // scratch GENERAL -> TRANSFER_SRC, dst TRANSFER_SRC -> TRANSFER_DST
    VkImageMemoryBarrier post[2] = {};
    post[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    post[0].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    post[0].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    post[0].newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[0].srcQueueFamilyIndex = post[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    post[0].image = o->scratchImg;
    post[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    post[1] = post[0];
    post[1].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    post[1].dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    post[1].newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    post[1].image = dst;
    d->CmdPipelineBarrier(cb,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_TRANSFER_BIT,
                          0, 0, nullptr, 0, nullptr, 2, post);

    VkImageCopy copyOut = {};
    copyOut.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyOut.dstSubresource = copyOut.srcSubresource;
    copyOut.extent = { w, h, 1 };
    d->CmdCopyImage(cb, o->scratchImg, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyOut);
}

// Lock held, the swapchain's work finished: its images' framebuffers go with it.
void OsdDestroySwapchain(DeviceData* d, SwapchainInfo& sc)
{
    OsdState* o = d->osd;
    if (!o) return;
    for (VkImage img : sc.images)
    {
        auto it = o->targets.find(img);
        if (it == o->targets.end()) continue;
        TargetDestroy(d, it->second);
        o->targets.erase(it);
    }
}

void OsdDestroyDevice(DeviceData* d)
{
    OsdState* o = d->osd;
    if (!o) return;
    // Compute path first (independent of the graphics render-pass teardown below).
    if (o->scratchView) d->DestroyImageView(d->device, o->scratchView, nullptr);
    if (o->scratchImg)  d->DestroyImage(d->device, o->scratchImg, nullptr);
    if (o->scratchMem)  d->FreeMemory(d->device, o->scratchMem, nullptr);
    if (o->cPipeline)       d->DestroyPipeline(d->device, o->cPipeline, nullptr);
    if (o->cPipelineLayout) d->DestroyPipelineLayout(d->device, o->cPipelineLayout, nullptr);
    if (o->cSetLayout)      d->DestroyDescriptorSetLayout(d->device, o->cSetLayout, nullptr);
    if (o->cPool)           d->DestroyDescriptorPool(d->device, o->cPool, nullptr);
    if (o->cs)              d->DestroyShaderModule(d->device, o->cs, nullptr);
    for (auto& kv : o->targets) TargetDestroy(d, kv.second);
    for (auto& kv : o->passes)
    {
        if (kv.second.pipeline) d->DestroyPipeline(d->device, kv.second.pipeline, nullptr);
        if (kv.second.pass) d->DestroyRenderPass(d->device, kv.second.pass, nullptr);
    }
    for (OsdSlot& s : o->slots) SlotDestroy(d, s);
    if (o->vs) d->DestroyShaderModule(d->device, o->vs, nullptr);
    if (o->fs) d->DestroyShaderModule(d->device, o->fs, nullptr);
    if (o->layout) d->DestroyPipelineLayout(d->device, o->layout, nullptr);
    if (o->pool) d->DestroyDescriptorPool(d->device, o->pool, nullptr);
    if (o->setLayout) d->DestroyDescriptorSetLayout(d->device, o->setLayout, nullptr);
    if (o->sampler && d->DestroySampler) d->DestroySampler(d->device, o->sampler, nullptr);
    for (HFONT f : { o->fontStatus, o->fontFps, o->fontHelp, o->fontHelpBody }) if (f) DeleteObject(f);
    delete o;
    d->osd = nullptr;
}

}} // namespace wiz3d::vk
