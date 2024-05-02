## AutoRDF2GML Example (Topology-based Node Features)

This example transforms the subsets of SemOpenAlex and AIFB RDF datasets:

1. semopenalex-C1793878-sample.nt, with its corresponding [config-soa-tb.ini](./config-soa-tb.ini) file,
2. [aifb-example-subset5k.nt](aifb-example-subset5k.nt) with its corresponding [config-aifb-tb.ini](./config-aifb-tb.ini) file.

### Configuration Steps

1. **Configure Paths**: Fill all the required fields in the config file, as in the following example:

    **[config-aifb-tb.ini](./config-aifb-tb.ini) Contents**:
    ```ini
    [InputPath] ;required
    input_path = aifb-example-subset5k.nt
    
    [SavePath] ;required
    save_path_numeric_graph = aifb-numeric-graph-complex/
    save_path_mapping = aifb-mapping-complex/
    
    [MODEL] ;required, options = transe / complex / distmult / rotate
    kge_model = complex
    
    [Nodes] ;required
    classes = person, group, publication
    person = http://swrc.ontoware.org/ontology#Person
    group = http://swrc.ontoware.org/ontology#ResearchGroup
    publication = http://swrc.ontoware.org/ontology#Publication
    ; pubtype = http://swrc.ontoware.org/ontology#
    
    [SimpleEdges] ;required
    edge_names = person_group, publication_person
    ```

### Running the Transformation

Execute the [autordf2gml-tb.py](./autordf2gml-tb.py) script to start the process, as below:

```python autordf2gml-cb.py --config_path config-cb.ini```, e.g.,

```python autordf2gml-cb.py --config_path config-aifb-tb.ini``` to run the transformation for AIFB dataset, and

```python autordf2gml-cb.py --config_path config-aifb-tb.ini``` to run the transformation for SemOpenAlex dataset.

The resulted GML dataset containing the numeric graph and mapping files will be saved in the output folder.
