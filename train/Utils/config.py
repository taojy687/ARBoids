import yaml
from types import SimpleNamespace

def _dict_to_namespace(d):
    '''
        Recursively convert dict to Dot-accessible Namespace.
    '''
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(x) for x in d]
    else:
        return d

def load_config(path: str):
    with open(path, "r") as f:
        raw_cfg = yaml.safe_load(f)
    return _dict_to_namespace(raw_cfg)

def _namespace_to_dict(ns):
    if isinstance(ns, dict):
        return {k: _namespace_to_dict(v) for k, v in ns.items()}
    if hasattr(ns, "__dict__"):
        return {k: _namespace_to_dict(v) for k, v in ns.__dict__.items()}
    if isinstance(ns, list):
        return [_namespace_to_dict(x) for x in ns]
    return ns