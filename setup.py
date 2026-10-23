from setuptools import setup

setup(
    name='hyperchamber',
    version='0.3.1',
    description='Tune and optimize your hyperparameters',
    author='255bits',
    author_email="support@255bits.com",
    packages=['hyperchamber', 'hyperchamber.io'],
    python_requires='>=3.11',
    install_requires=['requests>=2.33.0', 'certifi>=2026.10.19'],
    url='https://github.com/255bits/hyperchamber',
    license='MIT',
    classifiers=[
                 ],
)
