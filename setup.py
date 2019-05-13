from setuptools import setup

setup(
    name='horsekickerpy',
    version='0.1.0',
    packages=['horsekickerpy'],
    url='https://github.com/lstmemery/horsekickerpy',
    license='MIT',
    author='Matthew Emery',
    author_email='me@matthewemery.ca',
    description='Predict horse kick deaths at scale.',
    install_requires=['flask', 'pandas', 'scikit-learn', "flask_restplus"]
)
