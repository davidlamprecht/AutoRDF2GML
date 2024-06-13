### Step-by-step to package and publish the library to PyPI repository

#### (1) Prepare the code (in [autordf2gml](./autordf2gml) directory) and setup file ([pyproject.toml](./pyproject.toml))
Make sure the code is properly "callable", which means you need to describe the modules in the ``__init__.py`` file. This file should import the necessary components of the library so they are accessible when the library is used. The metadata for the package is written inside the [pyproject.toml](./pyproject.toml) file. Some key elements to include in this file are: 

``name``: This is the name of the library, which will be used with ``pip install name``,

``version``: This is important for managing updates to the library. When we upload a new version, we should increment this number. **IMPORTANT: Before making a new release, update the ``version`` field in [pyproject.toml](./pyproject.toml).**

#### (2) Build the distribution package. 

Make sure you are inside the [library](../library) folder where [pyproject.toml](./pyproject.toml) and [autordf2gml](./autordf2gml) directory exist. Then run the following script:

``python -m build``

Running the script would result in ``.egg-info`` and ``dist`` () directories; dist is the one containing the package distribution files (.tgz and wheel files).

#### (3) Publish the package

To publish it as a trial in test.pypi

``python -m twine upload --repository testpypi dist/*``

To publish it to the official PyPI repository

``python -m twine upload dist/*``

For comprehensive guide: https://packaging.python.org/en/latest/tutorials/packaging-projects/

P.S. Regarding versioning, there are several best practices we can follow. Semantic versioning is a common approach, where we use a version number in the format MAJOR.MINOR.PATCH (e.g., 1.0.0). This allows users to understand the nature of changes at a glance. --> according to GPT hahaha

