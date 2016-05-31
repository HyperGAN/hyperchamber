from setuptools import setup

setup(
    name='hyperchamber',
    version='0.1',
    description='Tune and optimize your hyperparameters',
    author='Martyn Garcia',
    author_email="martyn@255bits.com",
    packages=['hyperchamber'],
    install_requires=['requests >= 2.4.2'],
    url='https://github.com/255bits/hyperchamber',
    license='MIT',
    classifiers=['License :: MIT',
                 'Programming Language :: Python :: 3'],
)
