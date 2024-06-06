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
For the evaulation the edges are split to 80% for training, 10% for validation, and 10% for test. 
For training, we employ an early stop mechanism, determined by evaluating the validation loss of the validation set, to ensure complete training for all settings. 
We train a maximum of 100 epochs. All GNNs are implemented with 2 layers. We use 64 as the hidden dimension throughout all GNNs. 
For HGT, we set the head number as 8. We use Adam optimizer and set the learning rate to 0.001. 
As we formulate the recommendation task as a binary classification problem the loss function is binary cross-entropy.
We compute the loss by comparing the ground-truth labels with the obtained predictions.
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

### Bipartite Graph

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
