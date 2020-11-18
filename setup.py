from setuptools import setup
from os import path

req = open("requirements.txt", 'r').read()

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='seldonian',
    version='0.0.1dev',
    packages=['seldonian'],
    install_requires=req.splitlines(),
    url='https://abdulhannan.in/seldonian-fairness',
    license='MIT',
    author='Abdul Hannan Kanji',
    author_email='hannanabdul55@gmail.com',
    description='Train model with safety constraints',
    long_description=long_description,
    long_description_content_type='text/markdown'
)
