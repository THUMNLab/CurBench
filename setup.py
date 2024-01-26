import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CurBench",
    version="1.0.0",
    author="",
    author_email="",
    description="Curriculum Learning Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License"
    ],
    packages=setuptools.find_packages(exclude=("tests", "examples", "docs")),
    python_requires=">=3.7",
    install_requires=[],
)