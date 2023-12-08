## Example Use-Case for AutoRDF2GML

This example transforms the RDF knowledge graph [semopenalex-C1793878-sample.nt](./semopenalex-C1793878-sample.nt) into a heterogenous graph dataset creating content-based features for the nodes. SemOpenAlex-C1793878-sample.nt is a subset of [SemOpenAlex](https://semopenalex.org) containing works with the concept [*Out-of-order execution*](https://semopenalex.org/concept/C1793878).



For the transformation of the RDF data the following paths must be adjusted and then the [autordf2gml-cb.py](./autordf2gml-cb.py) script must be started: 

* #### [config-cb.ini](./config-cb.ini)

```
[InputPath]
input_path = /example/path/semopenalex-C1793878-sample.nt

[SavePath]
save_path_numeric_graph = /example/path/numeric-graph
save_path_mapping = /example/path/mapping
```

* #### [autordf2gml-cb.py](./autordf2gml-cb.py) 

```config.read('/example/path/config-cb.ini')``` (line 19)
