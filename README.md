# 🧩 AutoRDF2GML

**AutoRDF2GML** is a framework designed to convert RDF data into graph representations suitable for graph-based machine learning (GML) methods, such as Graph Neural Networks (GNNs). By generating both **content-based features** from RDF datatype properties and **topology-based features** from RDF object properties, AutoRDF2GML enables effective integration of Semantic Web technologies with Graph Machine Learning.

---

### 🌟 Key Features

- **Content-Based Node Features**: Automatically extract node features from RDF datatype properties.
- **Topology-Based Edge Features**: Derive edge features from RDF object properties.
- **User-Friendly Interface**: Modular design with automatic feature selection for simplicity and ease of use.
- **Graph ML Integration**: Seamlessly integrates with leading frameworks like PyTorch Geometric and DGL.

![Overview of AutoRDF2GML](autordf2gml-overview.png)

---

### 📥 Installation via pip

AutoRDF2GML is now available via pip! To install, simply run:
```bash
pip install autordf2gml
```
For detailed usage instructions, check [https://pypi.org/project/autordf2gml/](https://pypi.org/project/autordf2gml/).

---

## Quick User Guide

For a step-by-step guide on using the framework, see our [example](./example) and [example-topologyfeatures](./example/example-topologyfeatures) directories.

## Usage

To start using AutoRDF2GML, you need an **(1) RDF file** and **(2) config file** describing the configuration for the transformation. In the config file, define the RDF classes and properties as needed for your project. Once configured, execute the AutoRDF2GML script to generate a heterogeneous graph dataset suitable for your machine learning applications. For a step-by-step guide, see our [example](./example) and [example-topologyfeatures](./example/example-topologyfeatures) directories.

The output can then be used for various machine learning tasks, including node classification, link prediction, and graph classification. It can be readily integrated into common graph machine learning frameworks. For example, see how the output from AutoRDF2GML can be loaded into a PyTorch Geometric HeteroData object in this [script](./use-with-pyg/create-pyg-heterodata.py). For instance, the structure of the loaded PyG HeteroData object is available as a **directed** graph [here](./use-with-pyg/pyg-heterodata-soa-sw-directed.txt) and as an **undirected** graph [here](./use-with-pyg/pyg-heterodata-soa-sw-undirected.txt).

## Feature Configuration

### Content-based Node Features

**Quick example** for Content-based Node Features Transformation: [example](./example)

AutoRDF2GML with content-based node features is implemented in the Python script [autordf2gml-cb.py](./content-based-feature/autordf2gml-cb.py). The related template and documentation of the configuration file is defined in the [config-template.ini](./content-based-feature/config-template.ini) file.
The default model for calculating the embeddings based on the natural language descriptions is [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased), but also other huggingface BERT variant models (e.g., bert-base) can be used.

### Topology-based Node Features

**Quick example** for Topology-based Node Features Transformation: [example-topologyfeatures](./example/example-topologyfeatures) directory.

AutoRDF2GML with topology-based node features is implemented in the Python script [autordf2gml-tb.py](./topology-based-feature/autordf2gml-tb.py). The related template and documentation of the configuration file is defined in the [config-template.ini](./topology-based-feature/config-template.ini) file.
The following KG embedding models are possible for calculating the topology-based feature: [TransE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.TransE.html#torch_geometric.nn.kge.TransE), [DistMult](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.DistMult.html#torch_geometric.nn.kge.DistMult), [ComplEx](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.ComplEx.html#torch_geometric.nn.kge.ComplEx), [RotatE](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.kge.RotatE.html#torch_geometric.nn.kge.RotatE). The default parameters (hidden channel size 128) are defined and commented in the implementation. 

## 🤝 Contributing

We welcome any kind of contributions!

## 📄 License

**AutoRDF2GML** is available under the [MIT License](https://opensource.org/licenses/MIT), making it open and accessible for both personal and commercial use.

## GML Datasets

- **GML Dataset LPWC**  
  DOI: [10.5281/zenodo.10299366](https://doi.org/10.5281/zenodo.10299366)  
  License: CC BY-SA 4.0

- **GML Dataset SOA-SW**  
  DOI: [10.5281/zenodo.10299429](https://doi.org/10.5281/zenodo.10299429)  
  License: Creative Commons Zero (CC0)

- **GML Dataset AIFB**  
  DOI: [10.5281/zenodo.10989595](https://doi.org/10.5281/zenodo.10989595)  
  License: CC BY 4.0

- **GML Dataset LinkedMDB**  
  DOI: [10.5281/zenodo.10989683](https://doi.org/10.5281/zenodo.10989683)  
  License: CC BY 4.0

## 📞 Contact & Reference

Michael Färber, David Lamprecht, Yuni Susanti: ["AutoRDF2GML: Facilitating RDF Integration in Graph Machine Learning"](https://arxiv.org/pdf/2407.18735), Proceedings of the 23rd International Semantic Web Conference (ISWC'24), Baltimore, USA.
