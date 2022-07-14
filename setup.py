import setuptools
import subprocess

try:
    version = (
        subprocess.check_output(["git", "describe", "--abbrev=0", "--tags"])
        .strip()
        .decode("utf-8")
    )
except:
    print("Failed to retrieve the current version, defaulting to 0")
    version = "0"

setuptools.setup(
    name="imbalance",
    version=version,
    author="The CoCo Lab",
    description="An exploration of data imbalance",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[i.strip() for i in open("requirements.txt").readlines()],
    python_requires='==3.8.*',
)
