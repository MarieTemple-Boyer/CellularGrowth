""" Setup file """

import setuptools

setuptools.setup(
    # Includes all other files that are within your project folder
    include_package_data=True,
    # Name of your Package
    name='cellulargrowth',
    # Project Version
    version='1.0',
    # Description of your Package
    description='Solve an equation that models cellular development',
    # Website for your Project or Github repo
    url="https://github.com/MarieTemple-Boyer/CellularGrowth",
    # Name of the Creator
    author='Temple-Boyer Marie',
    # Creator's mail address
    author_email='temple-boyer.marie@orange.fr',
    # Projects you want to include in your Package
    packages=setuptools.find_packages(),
    # Dependencies/Other modules required for your package to work
    install_requires=['numpy'],
    # Detailed description of your package
    long_description='This module solve the equation found in the article \
                      from Perthame and Jabin (2022) with a diffusion of the nutrients.',
    # Format of your Detailed Description
    long_description_content_type="text/markdown",
    # Classifiers allow your Package to be categorized based on functionality
    classifiers=[
        "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
    ],
)
