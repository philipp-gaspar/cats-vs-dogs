from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'tensorflow>=2.0'
]

setup(
    name='cats_vs_dogs',
    version='0.1',
    author='Philipp Gaspar',
    author_email='philipp.gaspar@gmail.com',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    description='Cats vs dogs prediction in Cloud ML Engine.',
    requires=[]
)
