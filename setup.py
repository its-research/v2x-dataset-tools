from os.path import dirname, realpath
from setuptools import find_packages, setup

from v2x_datasets_tools.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='v2x_datasets_tools',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/its-research/v2x-dataset-tools',
    license='MIT',
    description=
    'An opensource dataset visualization framework for autonomous driving '
    'cooperative detection',
    long_description=open('README.md').read(),
    install_requires=_read_requirements_file(),
)
