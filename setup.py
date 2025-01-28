from setuptools import find_packages,setup
from typing import List

val = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path,'r') as file:
        requirements = file.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if val in requirements:
            requirements.remove(val)
    return requirements


setup(
    name='mlproject',
    version='0.0.1',
    author='Lakshay',
    author_email='lakshaylucky18@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)