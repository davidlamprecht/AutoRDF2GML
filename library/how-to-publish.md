### Step-by-step to package and publish the library to PyPI repository

#### (1) Prepare the code (``autordf2gml``) and configuration file (``pyproject.toml``)
Make sure the code is properly "callable", which means you need to describe the modules in the ``__init__.py`` file. This file should import the necessary components of your library so they are accessible when the library is used. The metadata for the package is written inside the ``pyproject.toml`` file. Some key elements to include in this file are: 

``name``: This is the name of the library, which will be used with ``pip install name``,

``version``: This is important for managing updates to your library. When you upload a new version, you should increment this number. 
**IMPORTANT: Before making a new release, update the ``version`` field in ``pyproject.toml``.**

#### (2) Build the distribution package. 

Make sure you are inside the ``/library`` folder where ``pyproject.toml`` exists:
Run the following script:
``python -m build``

would result in .egg-info and dist () directories; dist is the one containing the package (.tgz and wheel file)

#### (3) Publish the package

To publish it as a trial in test.pypi

``python -m twine upload --repository testpypi dist/*``

To publish it to the official PyPI repository

``python -m twine upload dist/*``

For comprehensive guide: https://packaging.python.org/en/latest/tutorials/packaging-projects/

P.S. Regarding versioning, there are several best practices we can follow. Semantic versioning is a common approach, where we use a version number in the format MAJOR.MINOR.PATCH (e.g., 1.0.0). This allows users to understand the nature of changes at a glance. --> according to GPT hahaha

