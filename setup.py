import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="continual-flame",
    version="0.0.1.21",
    author="Andrea Rosasco",
    author_email="andrearosasco.ar@gmail.com",
    description="A continual learning PyTorch package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andrew-r96/ContinualFlame",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)