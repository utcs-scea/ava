from nightwatch.model import Type, Argument, Function


# TODO: Split this function into it's three conceptual elements.
def timing_code_guest(selector: str, name: str, enabled: bool) -> str:
    if not enabled:
        return ""
    return f"""#ifdef AVA_PRINT_TIMESTAMP
        {{
            struct timeval ts_{name}_timing;
            gettimeofday(&ts_{name}_timing, NULL);
            printf("Guestlib: {name} {selector} at : %ld s, %ld us\\n", ts_{name}_timing.tv_sec, ts_{name}_timing.tv_usec);
        }}
    #endif\n"""


def report_type_alloc_resources(ty: Type) -> str:
    ret = ""
    for resource, amount in ty.allocates_resources.items():
        ret += f"""nw_report_storage_resource_allocation("{resource}", {amount});\n"""
    for resource, amount in ty.deallocates_resources.items():
        ret += f"""nw_report_storage_resource_allocation("{resource}", -{amount});\n"""
    return ret


def report_alloc_resources(arg: Argument) -> str:
    ret = ""
    for ty in arg.contained_types:
        ret += report_type_alloc_resources(ty)
    return ret


def report_consume_resources(f: Function) -> str:
    ret = ""
    for resource, amount in f.consumes_resources.items():
        ret += f"""nw_report_throughput_resource_consumption("{resource}", {amount});\n"""
    ret += f"""
    #ifdef AVA_API_FUNCTION_CALL_RESOURCE
        nw_report_throughput_resource_consumption("ava_api_function_call", 1);
    #endif
    """
    return ret


def timing_code_worker(location: str, name: str, enabled: bool) -> str:
    if not enabled:
        return "".strip()
    return f"""#ifdef AVA_PRINT_TIMESTAMP
        {{
            struct timeval ts_{name}_timing;
            gettimeofday(&ts_{name}_timing, NULL);
            printf("Worker: {name} {location} at : %ld s, %ld us\\n",ts_{name}_timing.tv_sec, ts_{name}_timing.tv_usec);
        }}
        #endif\n"""
