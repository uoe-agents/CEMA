import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(name='cema',
                 version='0.1.0',
                 description='Supporting code to the paper "Causal Social Explanations for '
                             'Stochastic Sequential Multi-Agent Decision-Making"',
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 author='Balint Gyevnar, Cheng Wang',
                 author_email='balint.gyevnar@ed.ac.uk',
                 url='https://github.com/uoe-agents/cema',
                 packages=setuptools.find_packages(exclude=["scenarios"]),
                 install_requires=requirements
                 )
