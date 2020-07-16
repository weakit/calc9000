from setuptools import setup, find_packages

setup(
    name='calc9000',
    version='0',
    packages=['calc9000'],
    python_requires='>=3.6',
    author='weakit',
    url='https://github.com/weakit/calc9000',
    download_url='https://github.com/weakit/calc9000/archive/master.tar.gz',
    install_requires=['sympy>=1.4', 'lark-parser'],
)
