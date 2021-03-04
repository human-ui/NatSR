from setuptools import setup, find_packages

setup(
    name='nat_sr',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'numpy',
        'imageio',
    ]
)
