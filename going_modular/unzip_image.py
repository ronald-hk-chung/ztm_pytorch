'''
Functions to Unzip --zippath to --target
'''
import os
import pathlib
import shutil
import zipfile
import argparse
import walk_through_dir

# Setting up the arg parser
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True)
parser.add_argument('--zippath', type=str, required=True)
args = parser.parse_args()

# Creating target directory
data_path = pathlib.Path(args.target)
data_path.mkdir(parents=True, exist_ok=True)
print(f'{data_path} created')

# Copying and zipref to target
shutil.copy2(src=str(args.zippath),
             dst=str(data_path / 'zipref.zip'))
print(f'Copied... {args.zippath} --> {data_path}')

with zipfile.ZipFile(data_path / 'zipref.zip') as zip_ref:
  zip_ref.extractall(data_path)
  print(f'Unzipped Files to {data_path}')

# Removing zipref
os.remove(data_path / 'zipref.zip')

# Walkthrough data_path
walk_through_dir.walk_through_dir(data_path)
