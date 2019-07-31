import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="intraCorr",
    version="0.0.1",
    author="Erika Munoz, Jose Nandez",
    author_email="erika.munoz.datascientist@gmail.com, jose.nandez.ds@gmail.com",
    description="This package calculates the correlation within subjects for repeated observations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JNandez/intraCorr",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=0.15',
        'pandas>=0.23',
        'scipy>=1.1',
        'matplotlib>=3.0',
        'sklearn>=0.19'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)
