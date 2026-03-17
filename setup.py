from setuptools import find_packages, setup
setup(
    name='football_oracle',
    version="0.1.0",
    packages=find_packages(),
    install_requires=[line.strip() for line in open("requirements.txt")],
    include_package_data=True
)
