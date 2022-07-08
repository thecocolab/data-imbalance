import setuptools

long_description = """Abstract of the paper."""
REQUIREMENTS = [i.strip() for i in open("requirements.txt").readlines()]
setuptools.setup(
    name="imbalanced_coconuts",
    version="0.0.1",
    author="The CoCo Lab",
    description="An exploration of data imbalance using cocos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires = REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)