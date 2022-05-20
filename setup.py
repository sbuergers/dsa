from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required_dependencies = f.read().splitlines()

setup(
    name='dsa',
    version='0.1.0',
    description='Data scientist associate certification learning materials',
    author='Steffen Burgers',
    license='',
    packages=find_packages(),
    install_requires=required_dependencies,
    zip_safe=False
)
