from typing import Callable, Dict, Any

__ops__ : Dict[str,Callable] = { }

def register_op(name: str):
    def inner_register(func : Callable):
        __ops__[name] = func
    return inner_register

def get_op(cls: Any) -> Callable:
    return __ops__[cls.__class__.__name__](cls)