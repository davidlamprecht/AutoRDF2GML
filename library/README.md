# AutoRDF2GML

## Overview

AutoRDF2GML is a framework designed to transform RDF data into graph representations suitable for graph-based machine learning methods, e.g., Graph Neural Networks (GNNs). It uniquely generates content-based features from RDF datatype properties and topology-based features from RDF object properties, enabling the effective integration of Semantic Web technologies with Graph Machine Learning.

## Installation

To install the current PyPI version, run:

```sh
pip install autordf2gml
```

We recommend users to use **isolated environment, such as venv or conda**, to use the library. Please note that the current version has only been tested with **Python versions 3.8 to 3.9.9**. 

## Usage

To start using AutoRDF2GML, you need: **(1) RDF file** and **(2) Configuration file** describing the configuration for the transformation. In the configuration file, define the RDF classes and properties as needed for your project. See the following for quick example. 

## Quick Example

This example uses the [semopenalex-C1793878-sample.nt](https://github.com/davidlamprecht/AutoRDF2GML/blob/main/example/semopenalex-C1793878-sample.nt) RDF file, a curated subset from [SemOpenAlex](https://semopenalex.org). 

#### 1. Preparing the configuration file

Fill all the required fields in the config file: see [config-soa-cb.ini](https://github.com/davidlamprecht/AutoRDF2GML/blob/main/example/config-soa-cb.ini) and [config-soa-tb.ini](https://github.com/davidlamprecht/AutoRDF2GML/blob/main/example/example-topologyfeatures/config-soa-tb.ini) as examples for the content-based and topology-based transformation, respectively. The following shows an example of the config file format:

    ```ini
    [InputPath] ;required
    input_path = semopenalex-C1793878-sample.nt

    [SavePath] ;required
    save_path_numeric_graph = semopenalex/numeric-graph/
    save_path_mapping = semopenalex/mapping/

    [NLD] ;required
    nld_class = work

    [EMBEDDING] ;required
    embedding_model = allenai/scibert_scivocab_uncased

    [Nodes] ;required
    classes = work, author, institution, source, concept, publisher
    work = https://semopenalex.org/class/Work
    author = https://semopenalex.org/class/Author
    institution = https://semopenalex.org/class/Institution
    
    [SimpleEdges] ;required
    edge_names = author_institution
    author_institution_start_node = author
    author_institution_properties = http://www.w3.org/ns/org#memberOf
    author_institution_end_node = institution
    ```

#### 2. Using the library

```python
import autordf2gml

#to run content-based transformation
autordf2gml.content_feature("config-soa-cb.ini") 

#to run topology-based transformation
autordf2gml.topology_feature("config-soa-tb.ini") 

#to run content-based transformation only using simple-edges
autordf2gml.simpleedges_feature("config-aifb-cb-simple.ini")
```

The required config files and RDF file for testing the library can be found in the [test](./test) directory. Simply run `python test.py` after installation to test the library.

## Our Github

The most recent updates, documentation, and examples can be accessed through the following repository:

- <https://github.com/davidlamprecht/AutoRDF2GML>
