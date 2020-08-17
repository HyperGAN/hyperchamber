from setuptools import setup

setup(
    name='hyperchamber',
    version='0.3.1',
    description='Tune and optimize your hyperparameters',
    author='255bits',
    author_email="support@255bits.com",
    packages=['hyperchamber', 'hyperchamber.io'],
    install_requires=['requests >= 2.4.2'],
    url='https://github.com/255bits/hyperchamber',
    license='MIT',
    classifiers=[
                 ],
)
