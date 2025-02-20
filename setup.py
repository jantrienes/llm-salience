import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="info_salience",
    version="0.0.1",
    author="Jan Trienes",
    author_email="jan.trienes@uni-marburg.de",
    description="info-salience",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jantrienes/info-salience",
    packages=setuptools.find_packages(where=["src"]),
    package_dir={"": "src"},
    python_requires=">=3.11",
)
