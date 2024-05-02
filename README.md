# AutoRDF2GML

AutoRDF2GML is an innovative framework designed to convert RDF data into graph representations suitable for graph-based machine learning methods such as Graph Neural Networks (GNNs). It uniquely generates content-based features from RDF datatype properties and topology-based features from RDF object properties, enabling the effective integration of Semantic Web technologies with Graph Machine Learning.


![Overview of AutoRDF2GML](autordf2gml-overview.png)

## Key Features

- **Content-based Node Features:** Automatically extract node features from RDF datatype properties.
- **Topology-based Edge Features:** Derive edge features from RDF object properties.
- **User-friendly Interface:** Features a modular design with automatic feature selection for simplicity and ease of use.
- **Graph ML Integration:** Seamlessly integrates with leading frameworks like PyTorch Geometric and DGL.

## Quick User Guide

For a step-by-step guide on using the framework, see our [example](./example) and [example-topologyfeatures](./example/example-topologyfeatures) directories.

## Usage

To start using AutoRDF2GML, you need an **(1) RDF file** and **(2) config file** describing the configuration for the transformation. In the config file, define the RDF classes and properties as needed for your project. Once configured, execute the AutoRDF2GML script to generate a heterogeneous graph dataset suitable for your machine learning applications. For a step-by-step guide, see our [example](./example) and [example-topologyfeatures](./example/example-topologyfeatures) directories.

The output can then be used for various machine learning tasks, including node classification, link prediction, and graph classification. It can be readily integrated into common graph machine learning frameworks. For example, see how the output from AutoRDF2GML can be loaded into a PyTorch Geometric HeteroData object in this [script](./use-with-pyg/create-pyg-heterodata.py). For instance, the structure of the loaded PyG HeteroData object is available as a **directed** graph [here](./use-with-pyg/pyg-heterodata-soa-sw-directed.txt) and as an **undirected** graph [here](./use-with-pyg/pyg-heterodata-soa-sw-undirected.txt).

## Feature Configuration

### Content-based Node Features

**Quick example** for Content-based Node Features Transformation can be found at the [example](./example) directory.

AutoRDF2GML with content-based node features is implemented in the Python script [autordf2gml-cb.py](./content-based-feature/autordf2gml-cb.py). The related template and documentation of the configuration file is defined in the [config-template.ini](./content-based-feature/config-template.ini) file.
The default model for calculating the embeddings based on the natural language descriptions is [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased), but also other huggingface BERT variant models (e.g., bert-base) can be used.

### Topology-based Node Features

**Quick example** for Topology-based Node Features Transformation can be found at the [example-topologyfeatures](./example/example-topologyfeatures) directory.

AutoRDF2GML with topology-based node features is implemented in the Python script [autordf2gml-tb.py](./topology-based-feature/autordf2gml-tb.py). The related template and documentation of the configuration file is defined in the [config-template.ini](./topology-based-feature/config-template.ini) file.
The following KG embedding models are possible for calculating the topology-based feature: [TransE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.TransE.html#torch_geometric.nn.kge.TransE), [DistMult](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.DistMult.html#torch_geometric.nn.kge.DistMult), [ComplEx](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.ComplEx.html#torch_geometric.nn.kge.ComplEx), [RotatE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.RotatE.html#torch_geometric.nn.kge.RotatE). The default parameters (hidden channel size 128) are defined and commented in the implementation. 

## Contributing

Contributions to AutoRDF2GML are welcome!

## License

AutoRDF2GML is made available under the [MIT License](https://opensource.org/licenses/MIT).
