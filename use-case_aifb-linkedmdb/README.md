## AutoRDF2GML Example (Content-based features, Binary relations)

An example to transform the AIFB RDF dataset by using only the **simple-edges (binary relations)** with **content-based features**.

### Step-by-step

1. **Configure paths**: Complete the configuration file [config-aifb-cb.ini](./config-aifb-cb.ini), as in the following example:

    ```ini
    [InputPath] ;input path
    input_path = path/to/RDFnt/file

    [SavePath] ;output paths for the numeric graph and mapping files
    save_path_numeric_graph = /path/to/output/numeric-graph
    save_path_mapping = /path/to/output/mapping

    [NLD] ;the node class correspond to the natural language description (NLD) class
    nld_class = publication

    [EMBEDDING] ;embeding model to create content-based features from NLD, model is BERT model variant from huggingface (bert-base, biobert, scibert, etc)
    embedding_model = allenai/scibert_scivocab_uncased

    [Nodes] ;nodes lists
    classes = person, group, publication
    person = http://swrc.ontoware.org/ontology#Person
    ...
    
    [SimpleEdges] ; binary relation lists
    edge_names = person_group, publication_person
    person_group_start_node = person
    person_group_properties = http://swrc.ontoware.org/ontology#affiliation
    ...
    ```

2. **Running the transformation**: Run the [autordf2gml-cb-simpleedges.py](./autordf2gml-cb-simpleedges.py), as below:

   ``` python autordf2gml-cb-simpleedges.py --config_path path/to/config ```

   **Example**: ``` python autordf2gml-cb-simpleedges.py --config_path config-aifb-cb.ini ``` 
