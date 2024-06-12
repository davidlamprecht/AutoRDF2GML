import configparser
import os

def load_config(config_path):
    print(f"## Loading the config file: {config_path}")
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def folder_check(mpath):
  if os.path.isdir(mpath): 
    print (f'## Path exists: {mpath}')
  else: 
    os.makedirs(mpath, exist_ok=True)
    print (f'## Path {mpath} created!')