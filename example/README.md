## AutoRDF2GML Example (Content-based Node Features)

This example uses the [semopenalex-C1793878-sample.nt](./semopenalex-C1793878-sample.nt) file, a curated subset from [SemOpenAlex](https://semopenalex.org) that contains works on the concept of *Out-of-order execution*.

### Configuration Steps

1. **Configure Paths**: Fill all the required fields in the config file: [config-soa-cb.ini](./config-soa-cb.ini), as in the following example:
   
    **[config-soa-cb.ini](./config-soa-cb.ini) Contents**:
    ```ini
    [InputPath]; required
   input_path = semopenalex-C1793878-sample.nt
   
   [SavePath]; required
   save_path_numeric_graph = semopenalex/numeric-graph/
   save_path_mapping = semopenalex/mapping/
    ```

### Running the Transformation

Execute the [autordf2gml-cb.py](./autordf2gml-cb.py) script to start the process, as below:

```python autordf2gml-cb.py --config_path path/to/config```, for example:

```python autordf2gml-cb.py --config_path config-soa-cb.ini```, to run the transformation for SemOpenAlex subset data.

## Conclusion

By following these steps, you have successfully transformed an RDF knowledge graph into an heterogenous graph dataset, ready to be used for GNNs.
