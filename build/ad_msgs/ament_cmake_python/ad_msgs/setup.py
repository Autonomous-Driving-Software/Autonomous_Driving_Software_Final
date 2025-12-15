from setuptools import find_packages
from setuptools import setup

setup(
    name='ad_msgs',
    version='0.0.0',
    packages=find_packages(
        include=('ad_msgs', 'ad_msgs.*')),
)
