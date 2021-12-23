def read_attr_conf(input_path,attr_name="config"):
    import importlib
    spec = importlib.util.spec_from_file_location("cfg",input_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod,attr_name)
