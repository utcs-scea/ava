from nightwatch.generator.c.stubs import function_implementation
from .command_handler import *
from nightwatch.model import API
from typing import Any, List, Tuple


def handle_call(api: API) -> str:
    return handle_command_function(api, list(api.real_functions) + list(api.callback_functions), api.callback_functions)


def source(api: API, errors: List[Any]) -> Tuple[str, str]:
    prelude = f"""
#define __AVA__ 1
#define ava_is_worker 1
#define ava_is_guest 0

#include "worker.h"

#undef AVA_BENCHMARKING_MIGRATE

{handle_command_header(api)}

void __attribute__((constructor(102))) init_{api.identifier.lower()}_worker(void) {{
    __handle_command_{api.identifier.lower()}_init();
    {api.worker_init_epilogue};
}}
"""
    stubs = f"""
////// API function stub implementations

#define __chan nw_global_command_channel

{lines(function_implementation(f) for f in api.callback_functions)}
"""

    function = handle_call(api)

    return api.c_worker_spelling, prelude + stubs + function
