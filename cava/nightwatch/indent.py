import subprocess

_indent_options = ("-nbad -bap -bc -bbo -hnl -br -brs -c50 -cd50 -ncdb -ce -ci4 -cli0 -d0 -di1 -nfc1 "
                   "-i4 -ip0 -l120 -nlp -npcs -nprs -psl -sai -saf -saw -ncs -nsc -sob -nfca -cp50 -ss "
                   "-ts8 -il1 -cbi0 -nut".strip().split())


def indent_c(code):
    try:
        with subprocess.Popen(["indent"] + _indent_options, encoding="utf-8" if hasattr(code, "encode") else None,
                              stdout=subprocess.PIPE, stdin=subprocess.PIPE) as proc:
            stdout, _ = proc.communicate(code)
            return stdout
    except FileNotFoundError:
        # Couldn't find indent, to just continue with raw code
        return code


def write_file_c(filename, data, indent=True, filename_prefix=""):
    with open(filename_prefix + filename, "w" if hasattr(data, "encode") else "wb") as fi:
        fi.write(indent_c(data) if indent else data)


def write_file_py(filename, data, filename_prefix=""):
    with open(filename_prefix + filename, "w" if hasattr(data, "encode") else "wb") as fi:
        fi.write(data)
