## AutoRDF2GML Example Use-Case

Welcome to this practical example on how to leverage AutoRDF2GML for transforming an RDF knowledge graph into a heterogenous graph dataset, enriching node features with content-based attributes.

### Getting Started

This example uses the [semopenalex-C1793878-sample.nt](./semopenalex-C1793878-sample.nt) file, a curated subset from [SemOpenAlex](https://semopenalex.org) that contains works on the concept of *Out-of-order execution*.


### Configuration Steps

1. **Configure Paths**: Adjust the paths in the [config-cb.ini](./config-cb.ini) file to point to the location of your `semopenalex-C1793878-sample.nt` file and your desired output locations for the numeric graph and mapping files.

    **[config-cb.ini](./config-cb.ini) Contents**:
    ```ini
    [InputPath]
   input_path = semopenalex-C1793878-sample.nt
   
   [SavePath]
   save_path_numeric_graph = semopenalex/numeric-graph/
   save_path_mapping = semopenalex/mapping/
    ```

### Running the Transformation

Execute the [autordf2gml-cb.py](./autordf2gml-cb.py) script to start the process, as below:

```python autordf2gml-cb.py --config_path config-cb.ini```

## Conclusion

By following these steps, you have successfully transformed an RDF knowledge graph into an heterogenous graph dataset, ready to be used for GNNs.
