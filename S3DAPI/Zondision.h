/*
* Project : wiz3D
* Copyright (C) wiz3D contributors 2026
*/

// Shared Zondision (Zimo M1 glasses-free display) output rail. Loads
// ZimoAdapter.dll at runtime and keeps the display's 3D state in sync with
// gInfo.Input.StereoActive each present. The on-screen image stays Half-SBS;
// the Zondision display itself weaves it through the 3DS SDK.
//
// Any wrapper that handles OUTMODE_ZONDISION (13) just needs to:
//   - call ZondisionTick() once per present after reading gInfo state
//   - call ZondisionShutdown() once at device teardown
// The rendering path is unchanged -- Zondision rides the normal Half-SBS blit.

#pragma once

#include "S3DAPI.h"   // S3DAPI_API export/import macro

namespace wiz3d {

// Parameterised core, used by wrappers that don't share S3DAPI's gInfo
// (S3DWrapperOGL has its own GlobalInfo instance, same for NvDirectMode and
// AmdQbProxy). Pass the wrapper's own outputMode + stereoActive values.
// Lazy-init on first call (loads ZimoAdapter.dll, picks a Zondision-registered
// display, starts a Game3DTransposer session in HARDWARE_WEAVING_SBS mode);
// keeps Set3D in sync on subsequent calls. Returns true when the SDK session is
// up and 3D-enabled.
S3DAPI_API bool ZondisionTickEx(unsigned outputMode, bool stereoActive);

// Convenience wrapper for callers that share S3DAPI's gInfo (DX7/8/9/10/11, VK).
S3DAPI_API bool ZondisionSelected();   // reads gInfo.OutputMode
S3DAPI_API bool ZondisionTick();       // reads gInfo.OutputMode + gInfo.Input.StereoActive

// Called once from each wrapper's device-teardown path. Safe to call without
// ZondisionTick() ever having run. Idempotent across wrappers in the same
// process (only the first call does the actual Stop + FreeLibrary).
S3DAPI_API void ZondisionShutdown();

} // namespace wiz3d
