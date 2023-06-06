from typing import *

def format_dict(info: dict, *, prefix: str="", end: str="\n"):
    n_key_char = max([len(s) for s in info.keys()])
    strings = ""
    for key, val in info.items():
        strings += f"{prefix}{key:<{n_key_char}} : "
        if isinstance(val, dict):
            strings += f"{end}" + format_dict(val, prefix=prefix+"  ")
        else:
            strings += f"{val}"
            strings += f"{end}"
    return strings

def print_info(info: dict, *, linewidth: int=60):
    print("=" * linewidth)
    print(format_dict(info), end="")
    print("=" * linewidth)
