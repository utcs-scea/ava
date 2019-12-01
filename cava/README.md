# CAvA

The internal code name is `nightwatch`.

The detailed documentation of design and usage is in [lapis.md](lapis.md).

## Requirement

* Python >= 3.6
* Python 3 Libraries: toposort, blessings, astor >= 0.7
* Native Libraries: libclang-7

### Installing requirements

* To install the Python version run the following:
```
sudo apt install python3.6
```
(A later version of Python 3 will also work. On non-debian system use the
appropriate package manager and package name.)
* To install the native dependencies run the following:
```
sudo apt install libclang-7-dev
```
* To install the Python dependencies run the following in the `nwcc` directory:
```
python3 -m pip install --user --requirement requirements.txt
```

## IDL Compiler generates:

* C Guest library (API stubs)
* Invoker (process that executes the called APIs in the appliance VM)

# Usage

Basic usage:
```
./nwcc [-h] [--language LANGUAGE] [-I INCLUDE_PATH] [-D DEFINITIONS]
       [-X EXTRA_ARGS] [-v] [-b] [-u] [--dump]
       SPEC_FILENAME
```

Run `nwcc --help` for additional options.
