# Evaluation of feature combinations of AutoRDF2GML datasets for Recommendation

We evaluate the performance of GNN models when being applied to the transformed graph machine learning datasets **soa-sw** and **lpwc**.


## GNN-based Recommendation Pipeline


## Semantic Feature  Initialization
AutoRDF2GML generates different kind of node features that can be used in various combinations. In the following, we outline different node feature initializations having different levels of semantic richness. Overall, we consider content-based and topology-based features as well as various combinations thereof:
1. One-hot-encoding (`one-hot`): As a foundational approach, we employ one-hot encoding for feature initialization.
2. Content-based: Natural language description (NLD, `cb_nld`): AutoRDF2GML with content-based node features but only 128-dimensional SciBERT embeddings from text attributes (natural language descriptions) are used.
3. Content-based: Literals (`cb_Literal`): AutoRDF2GML with content-based node feature
4. Topology-based (`tb`): AutoRDF2GML with topology-based node feature
5. NLD if available, otherwise topology-based (`comb_nld|tb`)
Combiantions:
6. Concatenation (`comb_Concat`): Concatenation of the Content-based and Topology-based Node Features
7. Addition (`comb_Addition`): Addition of the Content-based and Topology-based Node Features
8. Weighted Addition (`comb_WAddition`): Weighted Addition of the Content-based and Topology-based Node Features
9. Average (`comb_Average` ): Average of the Content-based and Topology-based Node Features
10. Neural Combinator (`comb_nc`): Neural combination via a feedforward neural network of the Content-based and Topology-based Node Features.


## Hyperparameter and Training. 
| Parameter                       | Value                                                |
|---------------------------------|------------------------------------------------------|
| Data Split                      | Training: 80%, Validation: 10%, Test: 10%            |
| Training Mechanism              | Early Stop based on validation loss                  |
| Maximum Epochs                  | 100                                                  |
| Number of GNN Layers            | 2                                                    |
| Hidden Dimension                | 64                                                   |
| Head Number (for HGT only)      | 8                                                    |
| Optimizer                       | Adam                                                 |
| Learning Rate                   | 0.001                                                |
| Loss Function                   | Binary Cross-Entropy                                 |
| Loss Calculation                | Comparing ground-truth labels with predictions       |


## Evaluation Results for GNN Models on SOA-SW

Evaluation results (F1 score, precision, recall, AUC score) of GNN models (GraphSAGE, GAT and HGT) with above mentioned feature initializations/combiantions and heterogeneity in the graph structure for paper recommendation (prediction of the edge work_author) on SOA-SW (best performances in bold).}


### Full Heterogenous Graph

| Feature Initialization      | GraphSAGE F1 | GraphSAGE Pre | GraphSAGE Re | GraphSAGE AUC | GAT F1 | GAT Pre | GAT Re | GAT AUC | HGT F1 | HGT Pre | HGT Re | HGT AUC |
|-----------------------------|--------------|---------------|--------------|---------------|--------|---------|--------|---------|--------|---------|--------|---------|
| `one-hot`                   | 0.806        | 0.899         | 0.731        | 0.937         | 0.875  | 0.925   | 0.830  | 0.962   | 0.890  | 0.880   | 0.901  | 0.949   |
| `cb_nld`                    | 0.874        | 0.932         | 0.823        | 0.967         | 0.877  | 0.924   | 0.834  | 0.961   | 0.886  | 0.901   | 0.872  | 0.957   |
| `cb_Literal`                | 0.914        | 0.927         | 0.901        | 0.972         | 0.889  | 0.919   | 0.861  | 0.964   | 0.887  | 0.882   | 0.892  | 0.945   |
| `tb`                        | 0.926        | 0.956         | 0.899        | 0.983         | 0.910  | 0.942   | 0.880  | 0.975   | 0.915  | 0.935   | 0.896  | 0.976   |
| `comb_nld|tb`               | 0.933        | 0.959         | 0.908        | 0.985         | 0.920  | 0.929   | 0.910  | 0.973   | 0.906  | 0.943   | 0.872  | 0.976   |
| `comb_Concat`               | 0.931        | 0.951         | 0.911        | 0.982         | 0.918  | 0.948   | 0.890  | 0.979   | 0.925  | **0.949** | 0.902  | **0.982** |
| `comb_Addition`             | 0.921        | 0.951         | 0.892        | 0.982         | 0.922  | **0.956** | 0.889  | 0.982   | 0.882  | 0.939   | 0.832  | 0.970   |
| `comb_WAddition`            | **0.940**    | 0.950         | **0.929**    | 0.984         | **0.923** | 0.954   | 0.894  | **0.983** | 0.885  | 0.934   | 0.841  | 0.968   |
| `comb_Average`              | 0.926        | **0.963**     | 0.893        | **0.987**     | 0.898  | 0.932   | 0.866  | 0.971   | **0.934** | 0.937   | **0.931** | 0.977   |
| `comb_nc`                   | 0.896        | 0.941         | 0.855        | 0.973         | 0.889  | 0.867   | **0.912** | 0.941   | 0.889  | 0.913   | 0.865  | 0.961   |

### Bipartite Graph (only the author and paper nodes are used)

| Feature Initialization      | GraphSAGE F1 | GraphSAGE Pre | GraphSAGE Re | GraphSAGE AUC | GAT F1 | GAT Pre | GAT Re | GAT AUC | HGT F1 | HGT Pre | HGT Re | HGT AUC |
|-----------------------------|--------------|---------------|--------------|---------------|--------|---------|--------|---------|--------|---------|--------|---------|
| `one-hot`                   | 0.810        | 0.893         | 0.740        | 0.855         | 0.823  | 0.894   | 0.763  | 0.863   | 0.824  | 0.896   | 0.763  | 0.850   |
| `cb_nld`                    | 0.855        | 0.911         | 0.805        | 0.942         | 0.830  | 0.871   | 0.793  | 0.892   | 0.854  | 0.836   | **0.873** | 0.924   |
| `cb_Literal`                | 0.882        | 0.910         | 0.856        | 0.955         | 0.846  | 0.852   | 0.841  | 0.903   | 0.847  | 0.846   | 0.848  | 0.914   |
| `tb`                        | **0.936**    | **0.969**     | 0.905        | **0.987**     | **0.895** | **0.914** | 0.877  | **0.952** | **0.892** | **0.940** | 0.850  | **0.967** |
| `comb_nld|tb`               | 0.905        | 0.904         | 0.905        | 0.965         | 0.872  | 0.877   | 0.866  | 0.928   | 0.828  | 0.898   | 0.768  | 0.915   |
| `comb_Concat`               | 0.922        | 0.936         | **0.908**    | 0.977         | 0.891  | 0.890   | **0.893** | 0.941   | 0.872  | 0.902   | 0.844  | 0.945   |
| `comb_Addition`             | 0.904        | 0.960         | 0.854        | 0.978         | 0.855  | 0.904   | 0.810  | 0.942   | 0.884  | 0.937   | 0.837  | 0.963   |
| `comb_WAddition`            | 0.910        | 0.956         | 0.869        | 0.977         | 0.873  | 0.902   | 0.845  | 0.939   | 0.876  | 0.904   | 0.850  | 0.949   |
| `comb_Average`              | 0.906        | 0.949         | 0.867        | 0.973         | 0.876  | 0.888   | 0.865  | 0.940   | 0.866  | 0.875   | 0.857  | 0.933   |
| `comb_nc`                   | 0.849        | 0.906         | 0.800        | 0.939         | 0.846  | 0.890   | 0.807  | 0.924   | 0.818  | 0.918   | 0.738  | 0.915   |


## Evaluation Results for GNN Models on LPWC
Evaluation results of GNN models (GraphSAGE, GAT and HGT) with above mentioned feature initializations/combiantions and
heterogeneity in the graph structure for task recommendation (prediction of the edge dataset_task) on LPWC (best perfor-
mances in bold).

## Evaluation Results for GNN Models on LPWC

### Full Heterogeneous Graph

| Feature Initialization      | GraphSAGE F1 | GraphSAGE Pre | GraphSAGE Re | GraphSAGE AUC | GAT F1 | GAT Pre | GAT Re | GAT AUC | HGT F1 | HGT Pre | HGT Re | HGT AUC |
|-----------------------------|--------------|---------------|--------------|---------------|--------|---------|--------|---------|--------|---------|--------|---------|
| `one-hot`                   | 0.739        | 0.791         | 0.694        | 0.834         | 0.748  | 0.766   | 0.732  | 0.794   | 0.778  | 0.801   | 0.756  | 0.859   |
| `cb_nld`                    | 0.837        | 0.873         | 0.803        | 0.928         | 0.802  | 0.742   | 0.872  | 0.847   | 0.783  | 0.721   | 0.857  | 0.862   |
| `cb_Literal`                | 0.784        | 0.697         | 0.895        | 0.853         | 0.800  | 0.834   | 0.769  | 0.879   | 0.820  | 0.778   | 0.868  | 0.894   |
| `tb`                        | 0.903        | 0.913         | 0.893        | 0.965         | 0.868  | **0.916** | 0.826  | 0.936   | 0.877  | 0.837   | 0.922  | 0.936   |
| `comb_Concat`               | 0.907        | 0.926         | 0.889        | 0.970         | 0.873  | 0.911   | 0.839  | 0.936   | 0.826  | 0.779   | 0.878  | 0.898   |
| `comb_Addition`             | 0.912        | 0.921         | 0.903        | 0.967         | 0.875  | 0.890   | 0.860  | 0.936   | **0.885** | **0.845** | **0.930** | **0.943** |
| `comb_WAddition`            | 0.918        | **0.933**     | 0.903        | **0.975**     | 0.872  | 0.896   | 0.849  | 0.938   | 0.875  | 0.841   | 0.912  | 0.936   |
| `comb_Average`              | **0.923**    | 0.920         | 0.926        | 0.971         | **0.882** | 0.882   | 0.883  | **0.942** | 0.829  | 0.767   | 0.903  | 0.896   |
| `comb_nc`                   | 0.879        | 0.832         | **0.932**    | 0.943         | 0.825  | 0.766   | **0.894** | 0.886   | 0.783  | 0.811   | 0.757  | 0.875   |

### Bipartite Graph

| Feature Initialization      | GraphSAGE F1 | GraphSAGE Pre | GraphSAGE Re | GraphSAGE AUC | GAT F1 | GAT Pre | GAT Re | GAT AUC | HGT F1 | HGT Pre | HGT Re | HGT AUC |
|-----------------------------|--------------|---------------|--------------|---------------|--------|---------|--------|---------|--------|---------|--------|---------|
| `one-hot`                   | 0.782        | 0.762         | 0.803        | 0.851         | 0.737  | 0.601   | **0.951** | 0.745   | 0.697  | 0.541   | 0.980  | 0.743   |
| `cb_nld`                    | 0.836        | 0.800         | 0.876        | 0.903         | 0.806  | 0.771   | 0.845  | 0.852   | 0.795  | 0.718   | 0.891  | 0.868   |
| `cb_Literal`                | 0.756        | 0.739         | 0.773        | 0.823         | 0.815  | 0.779   | 0.855  | 0.859   | 0.798  | 0.739   | 0.868  | 0.869   |
| `tb`                        | **0.915**    | 0.930         | 0.901        | 0.971         | **0.830** | **0.809** | 0.852  | **0.898** | 0.819  | **0.866** | 0.776  | **0.912** |
| `comb_Concat`               | 0.904        | **0.936**     | 0.873        | **0.973**     | 0.799  | 0.717   | 0.903  | 0.869   | **0.845** | 0.803   | 0.892  | 0.909   |
| `comb_Addition`             | 0.889        | 0.902         | 0.876        | 0.955         | 0.803  | 0.750   | 0.864  | 0.878   | 0.721  | 0.574   | **0.971** | 0.772   |
| `comb_WAddition`            | 0.882        | 0.836         | **0.933**    | 0.940         | 0.791  | 0.717   | 0.882  | 0.865   | 0.794  | 0.726   | 0.875  | 0.854   |
| `comb_Average`              | 0.862        | 0.885         | 0.841        | 0.940         | 0.793  | 0.726   | 0.872  | 0.869   | 0.732  | 0.599   | 0.939  | 0.769   |
| `comb_nc`                   | 0.813        | 0.746         | 0.895        | 0.875         | 0.740  | 0.617   | 0.923  | 0.725   | 0.740  | 0.603   | 0.958  | 0.767   |

