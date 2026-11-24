/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*/

// Shared Zondision output rail. See Zondision.h.
//
// The Zondision display's own driver weaves the on-screen image through the
// 3DS SDK (ZimoAdapter.dll -> 3DS_SDK.dll). wiz3D just produces a normal
// Half-SBS screen output and tells the SDK "3D on"; the SDK's weaving does
// the rest. If any step of the SDK path fails (DLL missing, no Zondision
// display, Start failed), the user still sees stereo on screen -- the output
// just isn't woven on the Zondision display.

#include "stdafx.h"
#include "Zondision.h"
#include "GlobalData.h"
#include "../Shared/OutputModeNames.h"
#include "../ZimoAdapter/ZimoAdapter.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <cstdio>

namespace {
// S3DAPI-local logger: OutputDebugStringA is universally available and lands in DbgView,
// VS Output window, and attached-debugger consoles. Avoids pulling in each wrapper's own
// Log* macro (DEBUG_MESSAGE, DDILog, etc. vary by project).
void ZLog(const char* fmt, ...)
{
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    _vsnprintf_s(buf, _countof(buf), _TRUNCATE, fmt, ap);
    va_end(ap);
    OutputDebugStringA(buf);
}
} // namespace

namespace wiz3d {

namespace {

struct ZondisionAdapter
{
    HMODULE                    dll = nullptr;
    Zimo_ApiVersionFn          ApiVersion = nullptr;
    Zimo_LoadFn                Load = nullptr;
    Zimo_ListDisplaysFn        ListDisplays = nullptr;
    Zimo_StartFn               Start = nullptr;
    Zimo_Set3DFn               Set3D = nullptr;
    Zimo_SetModeFn             SetMode = nullptr;
    Zimo_StopFn                Stop = nullptr;
    Zimo_PreferredDisplayFn    PreferredDisplay = nullptr;
    Zimo_DescribeFn            Describe = nullptr;
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
    bool             tried    = false;   // Init attempted (success or fail)
    bool             ready    = false;   // Session started
    bool             failed   = false;   // permanent fail: fall through to Half-SBS on screen
    int              displayId = 0;      // picked Zondision display
    bool             threeDOn = false;   // last Set3D state
};

ZondisionState& State()
{
    static ZondisionState s;
    return s;
}

} // namespace

bool ZondisionSelected()
{
    return gInfo.OutputMode == (DWORD)OUTMODE_ZONDISION;
}

bool ZondisionTick()
{
    return ZondisionTickEx((unsigned)gInfo.OutputMode, gInfo.Input.StereoActive != 0);
}

bool ZondisionTickEx(unsigned outputMode, bool stereoActive)
{
    if (outputMode != (unsigned)OUTMODE_ZONDISION) return false;
    ZondisionState& s = State();
    if (s.failed) return false;

    if (!s.tried)
    {
        s.tried = true;
        if (!LoadAdapter(s.adapter))
        {
            ZLog("Zondision: ZimoAdapter.dll missing or wrong API version; Zondision output unavailable\n");
            s.failed = true;
            return false;
        }
        if (const int r = s.adapter.Load())
        {
            ZLog("Zondision: 3DS_SDK.dll load failed (%s)\n", s.adapter.Describe(r));
            s.failed = true;
            return false;
        }
        ZimoDisplay displays[8] = {};
        const int n = s.adapter.ListDisplays(displays, 8);
        int picked = 0;
        const int pref = s.adapter.PreferredDisplay ? s.adapter.PreferredDisplay() : 0;
        for (int i = 0; i < n; ++i)
        {
            if (!displays[i].zondision) continue;              // Leia SR displays live on SR Weave rail
            if (pref && displays[i].id == pref) { picked = displays[i].id; break; }
            if (!picked) picked = displays[i].id;              // first zondision-registered display
        }
        if (!picked)
        {
            ZLog("Zondision: no Zondision-registered display found (SDK listed %d displays)\n", n);
            s.failed = true;
            return false;
        }
        s.displayId = picked;
        int features = 0;
        char sdkVer[64] = {};
        const int r = s.adapter.Start(picked, ZIMO_SBS, &features, sdkVer, (int)sizeof(sdkVer));
        if (r != 0)
        {
            ZLog("Zondision: Start(display=%d) failed: %s\n", picked, s.adapter.Describe(r));
            s.failed = true;
            return false;
        }
        ZLog("Zondision: session started on display %d (SDK %s, platformFeatures=%d)\n",
            picked, sdkVer, features);
        s.ready = true;
    }

    if (s.ready && !s.threeDOn && stereoActive)
    {
        if (const int r = s.adapter.Set3D(1))
            ZLog("Zondision: Set3D(1) failed: %s\n", s.adapter.Describe(r));
        else
            s.threeDOn = true;
    }
    else if (s.ready && s.threeDOn && !stereoActive)
    {
        s.adapter.Set3D(0);
        s.threeDOn = false;
    }
    return s.ready;
}

void ZondisionShutdown()
{
    ZondisionState& s = State();
    if (s.ready && s.adapter.Stop) s.adapter.Stop();
    if (s.adapter.dll) FreeLibrary(s.adapter.dll);
    s = ZondisionState{};
}

} // namespace wiz3d
