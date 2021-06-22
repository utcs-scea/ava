#include "guestlib/extensions/extension_api.h"

#include "common/extensions/tf_optimization.h"
#include "guestlib/guest_context.h"

/**
 * Initialization code in the generated code.
 */
void __helper_guestlib_init_prologue(ava::GuestContext *gctx) {
  ava_preload_cubin_guestlib();
  guestlib_tf_opt_init(gctx);
}

void __helper_guestlib_fini_prologue(ava::GuestContext *gctx) { guestlib_tf_opt_fini(gctx); }
