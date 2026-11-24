/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*/

// Zondision (Zimo M1 glasses-free display) output path for the Vulkan layer.
//
// Architecture mirrors the SR weave rail: Zondision's SDK does the per-pixel
// weave on the display itself when fed a Half-SBS image and told "3D on". We
// produce the Half-SBS via CompositeCompute (mode 1 in the compute shader) and
// drive the display state through ZimoAdapter.dll -- a tiny LGPL wrapper around
// 3DS_SDK.dll that holds the Zondision app key in XOR-scrambled form (built
// alongside wiz3D in ZimoAdapter/).
//
// Lifecycle:
//   - On first OUTMODE_ZONDISION present we LoadLibrary("ZimoAdapter.dll"), list
//     displays, pick the Zondision-registered one (or the SDK's preferred id),
//     and call Zimo_Start(id, ZIMO_SBS); Zimo_Set3D(On).
//   - If any step fails the path is marked failed and we fall through to the
//     normal Half-SBS present on the screen (so the user sees stereo, not black).
//   - On vkDestroyDevice we Zimo_Set3D(Off) + Zimo_Stop and FreeLibrary.

#include "Layer.h"
#include "../S3DAPI/GlobalData.h"
#include "../Shared/OutputModeNames.h"
#include "../ZimoAdapter/ZimoAdapter.h"

namespace wiz3d { namespace vk {

namespace {

struct ZondisionAdapter
{
    HMODULE               dll = nullptr;
    Zimo_ApiVersionFn     ApiVersion = nullptr;
    Zimo_LoadFn           Load = nullptr;
    Zimo_ListDisplaysFn   ListDisplays = nullptr;
    Zimo_StartFn          Start = nullptr;
    Zimo_Set3DFn          Set3D = nullptr;
    Zimo_SetModeFn        SetMode = nullptr;
    Zimo_StopFn           Stop = nullptr;
    Zimo_PreferredDisplayFn PreferredDisplay = nullptr;
    Zimo_DescribeFn       Describe = nullptr;
};

bool LoadAdapter(ZondisionAdapter& a)
{
    if (a.dll) return true;
    a.dll = LoadLibraryW(L"ZimoAdapter.dll");
    if (!a.dll) return false;
    a.ApiVersion       = (Zimo_ApiVersionFn)    GetProcAddress(a.dll, "Zimo_ApiVersion");
    a.Load             = (Zimo_LoadFn)          GetProcAddress(a.dll, "Zimo_Load");
    a.ListDisplays     = (Zimo_ListDisplaysFn)  GetProcAddress(a.dll, "Zimo_ListDisplays");
    a.Start            = (Zimo_StartFn)         GetProcAddress(a.dll, "Zimo_Start");
    a.Set3D            = (Zimo_Set3DFn)         GetProcAddress(a.dll, "Zimo_Set3D");
    a.SetMode          = (Zimo_SetModeFn)       GetProcAddress(a.dll, "Zimo_SetMode");
    a.Stop             = (Zimo_StopFn)          GetProcAddress(a.dll, "Zimo_Stop");
    a.PreferredDisplay = (Zimo_PreferredDisplayFn) GetProcAddress(a.dll, "Zimo_PreferredDisplay");
    a.Describe         = (Zimo_DescribeFn)      GetProcAddress(a.dll, "Zimo_Describe");
    const bool ok = a.ApiVersion && a.Load && a.ListDisplays && a.Start
                 && a.Set3D && a.Stop && a.Describe;
    if (!ok || a.ApiVersion() != ZIMO_ADAPTER_API_VERSION)
    {
        FreeLibrary(a.dll);
        a.dll = nullptr;
        return false;
    }
    return true;
}

struct ZondisionState
{
    ZondisionAdapter adapter;
    bool             tried   = false;   // Init attempted (success or fail)
    bool             ready   = false;   // Session started and 3D on
    bool             failed  = false;   // permanent fail: fall through to Half-SBS on screen
    int              displayId = 0;     // picked Zondision display, 0 = SDK's choice
    bool             threeDOn = false;  // last Set3D state (so we only call on change)
};

// Per-device singleton. Could live on DeviceData like OsdState / CompositeComputeState,
// but Zondision is single-session anyway (the SDK rejects a second Start). Simpler to
// scope here.
ZondisionState& State()
{
    static ZondisionState s;
    return s;
}

} // namespace

// Called the first present each frame that selects OUTMODE_ZONDISION. Returns true when
// the compute rail should produce Half-SBS output AND the Zondision 3D mode is on;
// false means fall through to the normal Half-SBS screen present (3D not available).
bool ZondisionBegin()
{
    if (gInfo.OutputMode != (DWORD)OUTMODE_ZONDISION) return false;
    ZondisionState& s = State();
    if (s.failed) return false;

    if (!s.tried)
    {
        s.tried = true;
        if (!LoadAdapter(s.adapter))
        {
            Log("Zondision: ZimoAdapter.dll missing or wrong API version; Zondision output unavailable\n");
            s.failed = true;
            return false;
        }
        if (const int r = s.adapter.Load())
        {
            Log("Zondision: 3DS_SDK.dll load failed (%s)\n", s.adapter.Describe(r));
            s.failed = true;
            return false;
        }
        ZimoDisplay displays[8] = {};
        const int n = s.adapter.ListDisplays(displays, 8);
        int picked = 0;
        const int pref = s.adapter.PreferredDisplay ? s.adapter.PreferredDisplay() : 0;
        for (int i = 0; i < n; ++i)
        {
            if (!displays[i].zondision) continue;              // Leia SR displays live on the SR Weave rail
            if (pref && displays[i].id == pref) { picked = displays[i].id; break; }
            if (!picked) picked = displays[i].id;              // first zondision-registered display
        }
        if (!picked)
        {
            Log("Zondision: no Zondision-registered display found (SDK listed %d displays)\n", n);
            s.failed = true;
            return false;
        }
        s.displayId = picked;
        int features = 0;
        char sdkVer[64] = {};
        const int r = s.adapter.Start(picked, ZIMO_SBS, &features, sdkVer, (int)sizeof(sdkVer));
        if (r != 0)
        {
            Log("Zondision: Start(display=%d) failed: %s\n", picked, s.adapter.Describe(r));
            s.failed = true;
            return false;
        }
        Log("Zondision: session started on display %d (SDK %s, platformFeatures=%d)\n",
            picked, sdkVer, features);
        s.ready = true;
    }

    if (s.ready && !s.threeDOn && gInfo.Input.StereoActive)
    {
        if (const int r = s.adapter.Set3D(1))
            Log("Zondision: Set3D(1) failed: %s\n", s.adapter.Describe(r));
        else
            s.threeDOn = true;
    }
    else if (s.ready && s.threeDOn && !gInfo.Input.StereoActive)
    {
        s.adapter.Set3D(0);
        s.threeDOn = false;
    }
    return s.ready;
}

bool ZondisionSelected()
{
    return gInfo.OutputMode == (DWORD)OUTMODE_ZONDISION;
}

void ZondisionShutdown()
{
    ZondisionState& s = State();
    if (s.ready && s.adapter.Stop) s.adapter.Stop();
    if (s.adapter.dll) FreeLibrary(s.adapter.dll);
    s = ZondisionState{};
}

}} // namespace wiz3d::vk
