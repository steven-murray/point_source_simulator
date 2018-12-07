from setuptools import setup
from distutils.core import Extension

import os
import sys
import re
import io

DEBUG = bool(os.getenv("DEBUG", False))

getvis = Extension(
    'pssim.get_visibilities',
    sources=['pssim/get_visibilities.c'],
    extra_compile_args=["-g -O0" if DEBUG else '-Ofast', '-fopenmp'],
    extra_link_args=['-lgomp']
)


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name="pssim",
    version=find_version("pssim", "__init__.py"),
    packages=['pssim'],
    install_requires=["numpy>=1.6.2", "spore", "scikit-learn", 'click'],
    author="Steven Murray",
    author_email="steven.g.murray@asu.edu",
    description="Simple point-source simulations of interferometric visibilities",
    long_description=read('README.rst'),
    license="MIT",
    keywords="power-spectrum signal processing",
    url="https://github.com/steven-murray/point_source_simulator",
    ext_modules=[getvis],
    entry_points={
        'console_scripts': [
            'run_layout = pssim.scripts.run_layout:main',
        ]
    }
    # could also include long_description, download_url, classifiers, etc.
)
