from typing import Iterable

from nightwatch.generator.c.callee import call_command_implementation
from nightwatch.generator.c.caller import return_command_implementation
from nightwatch.generator.c.printer import print_command_function
from nightwatch.generator.c.replay import replay_command_function
from nightwatch.generator.c.stubs import function_wrapper
from nightwatch.model import API, Function, lines


# TODO: Abstract the case structure of most functions into a class or something.


def handle_command_function(api: API, calls: Iterable[Function], returns: Iterable[Function]) -> str:
    function_name = f"__handle_command_{api.identifier.lower()}"
    calls = list(calls)
    returns = list(returns)
    return f"""
    {lines(function_wrapper(f) for f in calls)}

    void {function_name}_init() {{
        ava_endpoint_init(&__ava_endpoint, sizeof(struct {api.metadata_struct_spelling}), ava_is_worker ? 1 : 2);
        register_command_handler({api.number_spelling}, {function_name},
            __print_command_{api.identifier.lower()}, __replay_command_{api.identifier.lower()});
    }}

    void {function_name}_destroy() {{
        ava_endpoint_destroy(&__ava_endpoint);
    }}

    void {function_name}(struct command_channel* __chan, struct nw_handle_pool* handle_pool,
                         struct command_channel* __log, const struct command_base* __cmd) {{
        int ava_is_in, ava_is_out;
        switch (__cmd->command_id) {{
        {lines(return_command_implementation(f) for f in returns)}
        {lines(call_command_implementation(f, api.enabled_optimizations) for f in calls)}
        default:
            abort_with_reason("Received unsupported command");
        }} // switch
    }}

    {replay_command_function(api, calls)}

    {print_command_function(api)}
    """


def handle_command_header(api: API) -> str:
    return f"""
#include "common/endpoint_lib.hpp"
#include "common/linkage.h"

// Must be included before {api.c_header_spelling}, so that API
// functions are declared properly.
{api.include_lines}
#include "{api.c_header_spelling}"

#pragma GCC diagnostic ignored "-Wunused-function"

extern struct ava_endpoint __ava_endpoint;

static void __handle_command_{api.identifier.lower()}_init();
static void __handle_command_{api.identifier.lower()}_destroy();
void __replay_command_{api.identifier.lower()}(struct command_channel* __chan, struct nw_handle_pool* handle_pool,
                                    struct command_channel* __log,
                                    const struct command_base* __call_cmd, const struct command_base* __ret_cmd);
void __handle_command_{api.identifier.lower()}(struct command_channel* __chan, struct nw_handle_pool* handle_pool,
                                    struct command_channel* __log, const struct command_base* __cmd);
void __print_command_{api.identifier.lower()}(FILE* file, const struct command_channel* __chan,
                                    const struct command_base* __cmd);

#define ava_metadata(p) (&((struct {api.metadata_struct_spelling}*)ava_internal_metadata(&__ava_endpoint, p))->application)
// #define ava_zerocopy_alloc(s) ava_endpoint_zerocopy_alloc(&__ava_endpoint, s)
// #define ava_zerocopy_free(p) ava_endpoint_zerocopy_free(&__ava_endpoint, p)
// #define ava_zerocopy_get_physical_address(p) ava_endpoint_zerocopy_get_physical_address(&__ava_endpoint, p)


#include "{api.c_utilities_header_spelling}"

#ifndef NWCC_DEBUG

#ifndef __cplusplus
#pragma GCC diagnostic ignored "-Wincompatible-pointer-types"
#pragma GCC diagnostic ignored "-Wdiscarded-qualifiers"
#pragma GCC diagnostic ignored "-Wdiscarded-array-qualifiers"
#pragma GCC diagnostic ignored "-Wpointer-to-int-cast"
#endif

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#pragma GCC diagnostic ignored "-Wint-to-pointer-cast"
#pragma GCC diagnostic ignored "-Waddress"
#endif
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
#pragma GCC diagnostic ignored "-Wunused-variable"
    """
