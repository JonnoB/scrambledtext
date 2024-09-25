from setuptools import setup, find_packages

setup(
    name='scrambledtext',
    version='0.1',
    author = 'Jonathan Bourne',
    description='A library for creating synthetic corrupted OCR text using a markov process ',
    packages=find_packages(),
    include_package_data=True,
    package_data={'scrambledtext': ['corruption_distribs.json']},
    install_requires=[
        collections, re, random, math, json
    ],
)