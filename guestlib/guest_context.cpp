#include "guest_context.h"

#include <stdio.h>

#include "common/common_context.h"
#include "guestlib.h"

static auto common_context = ava::CommonContext::instance();
static auto guest_context = ava::GuestContext::instance();
