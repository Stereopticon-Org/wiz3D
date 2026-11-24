// ZimoKey.sample.h -- copy to ZimoKey.private.h (never committed, gitignored) and put
// wiz3D's Zondision app key in it to build ZimoAdapter.dll with that key. Without the
// private header, the adapter is built with Zondision's evaluation key (pre-scrambled
// in ZimoEvalKey.scrambled.inc; regenerate via tools/scramble_eval_key.py when it
// expires). Keys come from Zondision's developer portal (developer.zondision.com).
#pragma once
#define ZIMO_APP_KEY "your-app-key-here"
