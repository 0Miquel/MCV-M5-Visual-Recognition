import yaml
from yaml.loader import SafeLoader

def load_yaml_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=SafeLoader)
    return data