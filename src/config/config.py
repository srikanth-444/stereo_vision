import yaml

def load_config(config_file='config.yaml'):
    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)
    return config_data