import os

def mkdir_p(dirname):
    if not os.path.exists(dirname): os.makedirs(dirname)

def mkdir_from_filepath(filepath):
    dirname = os.path.dirname(filepath)
    mkdir_p(dirname)
