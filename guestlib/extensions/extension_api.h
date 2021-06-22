#ifndef _AVA_GUESTLIB_EXTENSIONS_EXTENSION_API_H_
#define _AVA_GUESTLIB_EXTENSIONS_EXTENSION_API_H_
namespace ava {
class GuestContext;
}
void ava_preload_cubin_guestlib();

void __helper_guestlib_init_prologue(ava::GuestContext *gctx);
void __helper_guestlib_fini_prologue(ava::GuestContext *gctx);

#endif  // _AVA_GUESTLIB_EXTENSIONS_EXTENSION_API_H_
