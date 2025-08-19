import yaml

def load_config(config_path='/mnt/data-1/users/why/quadruped_dog/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config