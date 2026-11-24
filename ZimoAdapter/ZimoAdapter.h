/*
* Project : wiz3D (adapted from Stereopticon-Org/SR-Loom, MIT-licensed; same interface)
* Copyright (C) wiz3D contributors 2026
*/

// ZimoAdapter.h -- the small C interface between wiz3D's Vulkan layer (and, in a
// later cycle, every DX wrapper) and ZimoAdapter.dll, which talks to a Zondision
// display through Zondision's 3DS SDK (3DS_SDK.dll, redistributed beside it).
//
// Why a separate DLL: Zondision's SDK identifies the calling application by an app
// key, and asks that an open-source project keep that key inside a binary rather
// than in a text file. The adapter is built with the key compiled in (lightly
// scrambled, see ZimoAdapter.cpp); wiz3D itself never sees it.
//
// wiz3D loads this DLL at runtime (LoadLibrary) and finds these entry points by name.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define ZIMO_ADAPTER_API_VERSION 2

typedef struct
{
    int  id;              // the SDK's display id (from 1)
    char name[32];        // the monitor's own name (EDID)
    int  width, height;   // native pixels
    int  mmWidth, mmHeight;
    int  zondision;       // a Zondision glasses-free display (ZLFD: the Zimo M1)
    int  sr;              // a Leia SR display (Zondision's SDK knows those too; wiz3D's SRWeave path owns those)
    char serial[16];
    int  shutter;         // a Zondision shutter-glasses monitor
    int  polarized;       // a Zondision polarised-glasses monitor
    int  ar;              // Zondision AR glasses
} ZimoDisplay;

typedef struct
{
    int   faces;          // faces seen (0: no viewer)
    float left[3];        // smoothed eye positions, display-centred, right-hand y-up (SDK units)
    float right[3];
    float headRotation[3];   // pitch, yaw, roll, degrees
    unsigned serial;      // goes up with each sample
} ZimoEyes;

// The adapter's interface version (ZIMO_ADAPTER_API_VERSION).
typedef int  (__cdecl* Zimo_ApiVersionFn)(void);
// Loads 3DS_SDK.dll (beside the adapter first, then the usual search). 0: loaded.
typedef int  (__cdecl* Zimo_LoadFn)(void);
// The 3D displays the SDK knows of: how many (up to max are filled in).
typedef int  (__cdecl* Zimo_ListDisplaysFn)(ZimoDisplay* out, int max);
// Weaving modes the adapter supports (mirrors the SDK's WeavingMode).
enum { ZIMO_SBS = 2, ZIMO_TB = 3, ZIMO_RGBDEPTH = 4, ZIMO_MULTIVIEW_TB = 5, ZIMO_AR = 6, ZIMO_POLARIZED = 7, ZIMO_SHUTTER = 8 };
// A session with display `id` (0: the SDK's choice) in `mode` (ZIMO_...). 0: started;
// else the SDK's error (Zimo_Describe). `features` (nullable) gets what the display can do,
// `sdkVersion` (nullable) the SDK's version text.
typedef int  (__cdecl* Zimo_StartFn)(int id, int mode, int* features, char* sdkVersion, int sdkVersionSize);
// Another mode for the running session (ZIMO_...): taken at once if 3D is on. 0: done.
typedef int  (__cdecl* Zimo_SetModeFn)(int mode);
// Multiview depth 0..62 (SDK default 16). 0: done.
typedef int  (__cdecl* Zimo_SetDepthFn)(int depth);
// The 3D display picked in Zondision's control panel (its id), 0 if none / unknown.
typedef int  (__cdecl* Zimo_PreferredDisplayFn)(void);
// 3D on (1) or off (0, 2D). 0: done, else the SDK's error.
typedef int  (__cdecl* Zimo_Set3DFn)(int on);
// The latest eye positions; 0 until the first.
typedef int  (__cdecl* Zimo_GetEyesFn)(ZimoEyes* out);
// Status the display sent since the last call, one at a time (the SDK's ACTLIST value and its number);
// 0 when there is none left.
typedef int  (__cdecl* Zimo_TakeStatusFn)(int* act, int* value);
// The session ended (2D first).
typedef void (__cdecl* Zimo_StopFn)(void);
// An SDK error number in words.
typedef const char* (__cdecl* Zimo_DescribeFn)(int ret);
// Which key the adapter was built with: 0 Zondision's evaluation key (expires), 1 wiz3D's own.
typedef int  (__cdecl* Zimo_KeyKindFn)(void);

#ifdef __cplusplus
}
#endif
