## Packaging library to publish to PyPI repository

This is a step-by-step summary, for comprehensive guide check: https://packaging.python.org/en/latest/tutorials/packaging-projects/


### (1) Prepare the code (in [autordf2gml](./autordf2gml) directory) and setup file ([pyproject.toml](./pyproject.toml))
Make sure the code is properly "callable", which means you need to describe the modules in the ``__init__.py`` file. This file should import the necessary components of the library so they are accessible when the library is used. The metadata for the package is written inside the [pyproject.toml](./pyproject.toml) file. Some key elements to include in this file are: 

``name``: This is the name of the library, which will be used with ``pip install name``,

``version``: This is important for managing updates to the library. When we upload a new version, we should increment this number. **IMPORTANT: Before making a new release, update the ``version`` field in [pyproject.toml](./pyproject.toml).**

``dependencies``: The dependencies needed to run the library, with their (compatible) versions.

``requires-python``: The python version, currently I put the cap at Python 3.8.0 to 3.9.9.

### (2) Build the distribution package

First upgrade the ``build`` library with ``python3 -m pip install --upgrade build``. Make sure you are in the same directory ([library](../library)) where [pyproject.toml](./pyproject.toml) and [autordf2gml](./autordf2gml) are located. Then run the following command:

``python -m build``

Running the command would result in ``.egg-info`` and ``dist`` directories; ``dist`` is the one containing the package distribution files (.tgz and wheel files).

### (3) Publish the package

First install the ``twine`` library: ``pip install --upgrade twine`` 

##### (a) To publish it as a trial in test.pypi

``python -m twine upload --repository testpypi dist/*``

The package is published in https://test.pypi.org/project/package-name. Then we can install it with: 

``pip install -i https://test.pypi.org/pypi/ --extra-index-url https://pypi.org/simple package-name --no-cache-dir``

##### (b) To publish it to the official PyPI repository

``python -m twine upload dist/*``

The package is published in https://pypi.org/project/autordf2gml. Then we can install it with: 

``pip install package-name``

Congratulations! You published the library!

P.S. Regarding versioning, there are several best practices we can follow. Semantic versioning is a common approach, where we use a version number in the format MAJOR.MINOR.PATCH (e.g., 1.0.0). This allows users to understand the nature of changes at a glance. --> according to GPT hahaha

