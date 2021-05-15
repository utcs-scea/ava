from nightwatch.generator.c.stubs import function_implementation, unsupported_function_implementation
from .command_handler import *
from nightwatch.model import API
from typing import Any, List, Tuple


def source(api: API, errors: List[Any]) -> Tuple[str, str]:
    handle_command_func_code = handle_command_function(
        api, api.callback_functions, list(api.real_functions) + list(api.callback_functions)
    )
    code = f"""
#define __AVA__ 1
#define ava_is_worker 0
#define ava_is_guest 1

#include "common/extensions/migration_barrier.h"
#include "guestlib.h"

{handle_command_header(api)}

void __attribute__((constructor(1))) init_{api.identifier.lower()}_guestlib(void) {{
    __handle_command_{api.identifier.lower()}_init();
    {api.guestlib_init_prologue};
    nw_init_guestlib({api.number_spelling});
    {api.guestlib_init_epilogue};
}}

void __attribute__((destructor)) destroy_{api.identifier.lower()}_guestlib(void) {{
    {api.guestlib_fini_prologue};
    nw_destroy_guestlib();
    {api.guestlib_fini_epilogue};
    __handle_command_{api.identifier.lower()}_destroy();
}}

{handle_command_func_code}

////// API function stub implementations

#define __chan nw_global_command_channel

{lines(function_implementation(f) for f in api.callback_functions)}
{lines(function_implementation(f) for f in api.real_functions)}
{lines(unsupported_function_implementation(f) for f in api.unsupported_functions)}

////// Replacement declarations

#define ava_begin_replacement 
#define ava_end_replacement 

{api.c_replacement_code}
    """.lstrip()
    return api.c_library_spelling, code
