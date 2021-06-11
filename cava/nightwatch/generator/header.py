from typing import Any, List, Tuple

from nightwatch import location, term, capture_errors, captured_errors

# pylint: disable=unused-import
from nightwatch.generator.common import _APISpelling
from nightwatch.model import API, Argument, Function, guard_macro_spelling, lines


def argument(arg: Argument) -> str:
    return f"""
    {arg.type.nonconst.attach_to(arg.param_spelling)};
    """.strip()


def function_call_struct(f: Function, errors: List[Any]):
    with capture_errors():
        with location(f"at {term.yellow(str(f.name))}", f.location, report_continue=errors):
            arg_suffix = "\n"
            return f"""
            struct {f.call_spelling} {{
                struct command_base base;
                intptr_t __call_id;
                {"".join(argument(a) + arg_suffix for a in f.arguments).strip()}
            }};
            """
        # noinspection PyUnreachableCode
        return f'#error "{captured_errors()}" '


def function_ret_struct(f: Function, errors: List[Any]):
    with capture_errors():
        with location(f"at {term.yellow(str(f.name))}", f.location, report_continue=errors):
            arg_suffix = "\n"
            return f"""
            struct {f.ret_spelling} {{
                struct command_base base;
                intptr_t __call_id;
                {"".join(argument(a) + arg_suffix for a in f.arguments if a.type.contains_buffer and a.output).strip()}\
                {argument(f.return_value) if not f.return_value.type.is_void else ""}
            }};
            """
        # noinspection PyUnreachableCode
        return f'#error "{captured_errors()}" '


def function_call_record_struct(f: Function, errors: List[Any]):
    with capture_errors():
        with location(f"at {term.yellow(str(f.name))}", f.location, report_continue=errors):
            arg_suffix = "\n"
            return f"""
            struct {f.call_record_spelling} {{
                {"".join(argument(a) + arg_suffix for a in f.arguments).strip()}\
                {argument(f.return_value) if not f.return_value.type.is_void else ""}\
                {"".join(argument(a) + arg_suffix for a in f.logue_declarations).strip()}\
                char __handler_deallocate;
                volatile char __call_complete;
            }};
            """
        # noinspection PyUnreachableCode
        return f'#error "{captured_errors()}" '


def function_struct(f: Function, errors: List[Any]) -> str:
    return function_call_struct(f, errors) + function_ret_struct(f, errors) + function_call_record_struct(f, errors)


def command_ids(f: Function) -> str:
    return f"{f.call_id_spelling}, {f.ret_id_spelling}"


def header(api: API, errors: List[Any]) -> Tuple[str, str]:
    functions = list(api.supported_functions)
    # TODO: Any objects pointed to by metadata will be leaked when the metadata is discarded.
    #   Metadata needs a destructor.
    code = f"""
#ifndef {guard_macro_spelling(api.c_header_spelling)}
#define {guard_macro_spelling(api.c_header_spelling)}

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <glib.h>

#include "common/cmd_channel.hpp"

#include "{api.c_types_header_spelling}"

#define {api.number_spelling} {api.number}

enum {api.functions_enum_spelling} {{
    {", ".join(command_ids(f) for f in functions)}
}};

#include "{api.c_utility_types_header_spelling}"

// These functions are required for timer code.
static inline void tvsub(struct timeval *x, struct timeval *y,
                         struct timeval *out)
{{
    out->tv_sec = x->tv_sec - y->tv_sec;
    out->tv_usec = x->tv_usec - y->tv_usec;
    if (out->tv_usec < 0) {{
        out->tv_sec--;
        out->tv_usec += 1000000;
    }}
}}

struct timestamp {{
    struct timeval start;
    struct timeval end;
}};

void probe_time_start(struct timestamp *ts)
{{
    gettimeofday(&ts->start, NULL);
}}

float probe_time_end(struct timestamp *ts)
{{
    struct timeval tv;
    gettimeofday(&ts->end, NULL);
    tvsub(&ts->end, &ts->start, &tv);
    return (tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0);
}}

struct {api.metadata_struct_spelling} {{
    struct ava_metadata_base base;
    {api.metadata_type.spelling if api.metadata_type else "int"} application;
}};

{lines(function_struct(f, errors) for f in functions)}

#endif // ndef {guard_macro_spelling(api.c_header_spelling)}
""".lstrip()
    return api.c_header_spelling, code


def utilities_header(api: API) -> Tuple[str, str]:
    code = f"""
#ifndef {guard_macro_spelling(api.c_utilities_header_spelling)}
#define {guard_macro_spelling(api.c_utilities_header_spelling)}

#define ava_utility static
#define ava_begin_utility 
#define ava_end_utility 
{api.c_utility_code.strip()}
#undef ava_utility
#undef ava_begin_utility 
#undef ava_end_utility 

#endif // ndef {guard_macro_spelling(api.c_utilities_header_spelling)}
""".lstrip()
    return api.c_utilities_header_spelling, code


def utility_types_header(api: API) -> Tuple[str, str]:
    code = f"""
#ifndef {guard_macro_spelling(api.c_utility_types_header_spelling)}
#define {guard_macro_spelling(api.c_utility_types_header_spelling)}

{api.c_type_code.strip()}

#endif // ndef {guard_macro_spelling(api.c_utility_types_header_spelling)}
""".lstrip()
    return api.c_utility_types_header_spelling, code


def types_header(api: API) -> Tuple[str, str]:
    nw_header = f"""
/** NightWatch MODIFIED header for {api.name} (version {api.version})
 *  Lines marked with "NWR:" were removed by NightWatch.
 *  All lines are from the original headers: {", ".join(api.includes)}
 */
""".lstrip()
    return api.c_types_header_spelling, nw_header + api.c_types_header_code
