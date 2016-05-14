from setuptools import setup

setup(
    name='hyperchamber',
    version='0.1',
    description='Tune and optimize your hyperparameters',
    author='Martyn Garcia',
    author_email="martyn@255bits.com",
    packages=['hyperchamber'],
    install_requires=['tensorflow>=0.7.0', 'numpy>=1.7', 'scipy>=0.16'],
    url='https://github.com/255bits/hyperchamber',
    license='MIT',
    classifiers=['License :: MIT',
                 'Programming Language :: Python :: 3'],
)
