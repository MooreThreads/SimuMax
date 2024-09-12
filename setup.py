"""Package Setup"""
import os
import re
from distutils.core import setup

CURRENT_DIR = os.path.dirname(__file__)

def read(path):
    with open(path, "r") as filep:
        return filep.read()

def get_version(package_name):
    if os.getenv("TAG_NAME"):
        return os.getenv("TAG_NAME")
    with open(os.path.join(CURRENT_DIR, package_name, "version.py")) as fp:
        for line in fp:
            tokens = re.search(r'^\s*__version__\s*=\s*"(.+)"\s*$', line)
            if tokens:
                return tokens.group(1)
    raise RuntimeError("No version found!")


setup(
    name='simumax',
    version=get_version("simumax"),
    url="https://github.com/MooreThreads/SimuMax",
    description="SimuMax: a static analytical model for LLM distributed training",
    long_description=read(os.path.join(CURRENT_DIR, "README.md")),
    long_description_content_type="text/markdown",
    author='MT AI Team',
    author_email='yutian.rong@mthreads.com',
    packages=['simumax'],
    install_requires=[]
)
