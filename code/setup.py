""" setup file """

import setuptools

setuptools.setup(
    name="code_covid_models",
    version="0.0.1",
    author="Juan Saez Hidalgo",
    author_email="juan.saez.hidalgo@gmail.com",
    description="Code for Covid models",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.8",
)
