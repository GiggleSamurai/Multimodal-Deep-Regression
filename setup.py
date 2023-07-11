from setuptools import setup, find_packages

# Requirements from requirements.txt
with open('requirements.txt', 'r') as file:
    requirements = file.read().splitlines()

setup(
    name='multi-deep-regression',
    version='1.0',
    author='Louis Wong, Ahmed Salih, Jason Xu, Mingyao Song',
    description="""Multi-Deep-Regression""",
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.8'
)
