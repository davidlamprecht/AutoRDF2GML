## AutoRDF2GML Example (Content-based features, Binary relations)

### Getting Started

An example to transform the AIFB RDF dataset by using only the **simple-edges (binary relations)** with **content-based features**.


### Configuration file

1. **Configure Paths**: Complete the configuration file [config-aifb-cb.ini](./config-aifb-cb.ini), i.e., define the node classes and edges, adjust the input/output paths, the node class correspond to the natural language description (NLD) class, and the embedding model (Huggingface BERT variant models e.g., bert-base, biobert, scibert, etc).

    ```ini
    [InputPath]
    input_path = path/to/RDFnt/file

    [SavePath]
    save_path_numeric_graph = /path/to/output/numeric-graph
    save_path_mapping = /path/to/output/mapping

    [NLD]
    nld_class = publication

    [EMBEDDING]
    embedding_model = allenai/scibert_scivocab_uncased
    ```

### Running the Transformation

Run the [autordf2gml-cb-simpleedges.py](./autordf2gml-cb-simpleedges.py), as below:

``` python autordf2gml-cb-simpleedges.py --config_path path/to/configfile ``` 

e.g.,

``` python autordf2gml-cb-simpleedges.py --config_path config-aifb-cb.ini ``` 
