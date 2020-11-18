from setuptools import setup
from os import path

req = open("requirements-release.txt", 'r').read()

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='seldonian',
    version='0.0.2dev',
    packages=['seldonian'],
    install_requires=req.splitlines(),
    python_requires='>3.5',
    url='https://abdulhannan.in/seldonian-fairness',
    license='MIT',
    author='Abdul Hannan Kanji',
    author_email='kanji@adulhannan.in',
    description='Train model with safety constraints',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
