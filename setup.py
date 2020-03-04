#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup
import shlex
import subprocess


def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode()


version = git_version()

setup(
    name='gbp',
    version=version,
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'networkx==2.2'
    ],
    author='Joseph Ortiz',
    author_email='joeaortiz16@gmail.com',
)
