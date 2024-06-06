from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, to_hetero
from torch_geometric.profile import count_parameters
from torch_geometric.loader import LinkNeighborLoader
import torch_geometric.transforms as T
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score 
import torch.nn.functional as F
from torch import Tensor
import torch
import pandas as pd
import numpy as np
import sys
import tqdm


#summary text file with evaluation results
orig_stdout = sys.stdout
f = open('eval-soa-sw-aw/gat/results/06_eval-gat-combined-concatenated.txt', 'w')
sys.stdout = f

seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

#load data

#lit
work_csv = "soa-sw/idx/nodes-literals/work_literals.csv"
author_csv = "soa-sw/idx/nodes-literals/author_literals.csv"
publisher_csv = "soa-sw/idx/nodes-literals/publisher_literals.csv"
source_csv = "soa-sw/idx/nodes-literals/source_literals.csv"
concept_csv = "soa-sw/idx/nodes-literals/concept_literals.csv"
institution_csv = "soa-sw/idx/nodes-literals/institution_literals.csv"

#transe embeddings
work_embeddings_csv = "soa-sw/idx/nodes-transe/work_transe.csv"
author_embeddings_csv = "soa-sw/idx/nodes-transe/author_transe.csv"
publisher_embeddings_csv = "soa-sw/idx/nodes-transe/publisher_transe.csv"
institution_embeddings_csv = "soa-sw/idx/nodes-transe/institution_transe.csv"
source_embeddings_csv = "soa-sw/idx/nodes-transe/source_transe.csv"
concept_embeddings_csv = "soa-sw/idx/nodes-transe/concept_transe.csv"

work_embeddings_df = pd.read_csv(work_embeddings_csv, header=None)
author_embeddings_df = pd.read_csv(author_embeddings_csv, header=None)
publisher_embeddings_df = pd.read_csv(publisher_embeddings_csv, header=None)
institution_embeddings_df = pd.read_csv(institution_embeddings_csv, header=None)
source_embeddings_df = pd.read_csv(source_embeddings_csv, header=None)
concept_embeddings_df = pd.read_csv(concept_embeddings_csv, header=None)

work_df = pd.read_csv(work_csv, header=None)
author_df = pd.read_csv(author_csv, header=None)
publisher_df = pd.read_csv(publisher_csv, header=None)
source_df = pd.read_csv(source_csv, header=None)
concept_df = pd.read_csv(concept_csv, header=None)
institution_df = pd.read_csv(institution_csv, header=None)

work_df = work_df.astype(float)
author_df = author_df.astype(float)
publisher_df = publisher_df.astype(float)
source_df = source_df.astype(float)
concept_df = concept_df.astype(float)
institution_df = institution_df.astype(float)

has_author_csv = "soa-sw/idx/edges/work_author.csv"
has_publisher_csv = "soa-sw/idx/edges/source_publisher.csv"
has_source_csv = "soa-sw/idx/edges/work_source.csv"
has_work = "soa-sw/idx/edges/work_work.csv"
has_concept = "soa-sw/idx/edges/work_concept.csv"
has_institution = "soa-sw/idx/edges/author_institution.csv"
has_coauthor = "soa-sw/idx/edges/author_author.csv"

has_author_df = pd.read_csv(has_author_csv, header=None)
has_publisher_df = pd.read_csv(has_publisher_csv, header=None)
has_source_df = pd.read_csv(has_source_csv, header=None)
has_work_df = pd.read_csv(has_work, header=None)
has_concept_df = pd.read_csv(has_concept, header=None)
has_institution_df = pd.read_csv(has_institution, header=None)
has_coauthor_df = pd.read_csv(has_coauthor, header=None)

#concatenate node features
combined_array_work = np.concatenate([work_df.values, work_embeddings_df.values], axis=1)
combined_array_author = np.concatenate([author_df.values, author_embeddings_df.values], axis=1)
combined_array_publisher = np.concatenate([publisher_df.values, publisher_embeddings_df.values], axis=1)
combined_array_institution = np.concatenate([institution_df.values, institution_embeddings_df.values], axis=1)
combined_array_source = np.concatenate([source_df.values, source_embeddings_df.values], axis=1)
combined_array_concept = np.concatenate([concept_df.values, concept_embeddings_df.values], axis=1)

work_tensor = torch.tensor(combined_array_work, dtype=torch.float)
author_tensor = torch.tensor(combined_array_author, dtype=torch.float)
publisher_tensor = torch.tensor(combined_array_publisher, dtype=torch.float)
institution_tensor = torch.tensor(combined_array_institution, dtype=torch.float)
source_tensor = torch.tensor(combined_array_source, dtype=torch.float)
concept_tensor = torch.tensor(combined_array_concept, dtype=torch.float)

has_author_src = torch.tensor(has_author_df.iloc[:, 0].values, dtype=torch.long)
has_author_dst = torch.tensor(has_author_df.iloc[:, 1].values, dtype=torch.long)
has_publisher_src = torch.tensor(has_publisher_df.iloc[:, 0].values, dtype=torch.long)
has_publisher_dst = torch.tensor(has_publisher_df.iloc[:, 1].values, dtype=torch.long)
has_source_src = torch.tensor(has_source_df.iloc[:, 0].values, dtype=torch.long)
has_source_dst = torch.tensor(has_source_df.iloc[:, 1].values, dtype=torch.long)
has_work_src = torch.tensor(has_work_df.iloc[:, 0].values, dtype=torch.long)
has_work_dst = torch.tensor(has_work_df.iloc[:, 1].values, dtype=torch.long)
has_concept_src = torch.tensor(has_concept_df.iloc[:, 0].values, dtype=torch.long)
has_concept_dst = torch.tensor(has_concept_df.iloc[:, 1].values, dtype=torch.long)
has_institution_src = torch.tensor(has_institution_df.iloc[:, 0].values, dtype=torch.long)
has_institution_dst = torch.tensor(has_institution_df.iloc[:, 1].values, dtype=torch.long)
has_coauthor_src = torch.tensor(has_coauthor_df.iloc[:, 0].values, dtype=torch.long)
has_coauthor_dst = torch.tensor(has_coauthor_df.iloc[:, 1].values, dtype=torch.long)

data = HeteroData()

data['work'].node_id = torch.arange(len(work_tensor))
data['author'].node_id = torch.arange(len(author_tensor))
data['publisher'].node_id = torch.arange(len(publisher_tensor))
data['source'].node_id = torch.arange(len(source_tensor))
data['concept'].node_id = torch.arange(len(concept_tensor))
data['institution'].node_id = torch.arange(len(institution_tensor))

data['work'].x = work_tensor
data['author'].x = author_tensor
data['publisher'].x = publisher_tensor
data['source'].x = source_tensor
data['concept'].x = concept_tensor
data['institution'].x = institution_tensor

#changed edge direction
data['author', 'has_work', 'work'].edge_index = torch.stack([has_author_dst, has_author_src], dim=0)

#not changed edge direction
data['work', 'has_source', 'source'].edge_index = torch.stack([has_source_src, has_source_dst], dim=0)
data['source', 'has_publisher', 'publisher'].edge_index = torch.stack([has_publisher_src, has_publisher_dst], dim=0)
data['work', 'has_work', 'work'].edge_index = torch.stack([has_work_src, has_work_dst], dim=0)
data['work', 'has_concept', 'concept'].edge_index = torch.stack([has_concept_src, has_concept_dst], dim=0)
data['author', 'has_institution', 'institution'].edge_index = torch.stack([has_institution_src, has_institution_dst], dim=0)
data['author', 'has_coauthor', 'author'].edge_index = torch.stack([has_coauthor_src, has_coauthor_dst], dim=0)

#lin layer edge features
has_author_lin = torch.nn.Linear(3, 1)
has_concept_lin = torch.nn.Linear(1, 1)
has_coauthor_lin = torch.nn.Linear(1, 1) 

has_author_features_csv = "numeric-soa-sw/idx/edges/feature_work_author.csv"
has_author_features_df = pd.read_csv(has_author_features_csv, header=None)
has_author_features_tensor = torch.tensor(has_author_features_df.values, dtype=torch.float)
data['author', 'has_work', 'work'].edge_attr = has_author_lin(has_author_features_tensor)

has_concept_features_csv = "numeric-soa-sw/idx/edges/feature_work_concept.csv"
has_concept_features_df = pd.read_csv(has_concept_features_csv, header=None)
has_concept_features_tensor = torch.tensor(has_concept_features_df.values, dtype=torch.float)
data['work', 'has_concept', 'concept'].edge_attr = has_concept_lin(has_concept_features_tensor)

has_coauthor_features_csv = "numeric-soa-sw/idx/edges/feature_author_author.csv"
has_coauthor_features_df = pd.read_csv(has_coauthor_features_csv, header=None)
has_coauthor_features_tensor = torch.tensor(has_coauthor_features_df.values, dtype=torch.float)
data['author', 'has_coauthor', 'author'].edge_attr = has_coauthor_lin(has_coauthor_features_tensor)

#finished loading data

data = T.ToUndirected()(data)

#train, val, test split
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=[('author', 'has_work', 'work')],
    rev_edge_types=[('work', 'rev_has_work', 'author')], 
)
train_data, val_data, test_data = transform(data)


# Define the train seed edges:
edge_label_index_train = train_data["author", "has_work", "work"].edge_label_index
edge_label_train = train_data["author", "has_work", "work"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[100, 50],
    neg_sampling_ratio=1.0,
    edge_label_index=(("author", "has_work", "work"), edge_label_index_train),
    edge_label=edge_label_train,
    batch_size= 2048 ,
    shuffle=True,
)

# Define the validation seed edges:
edge_label_index_val = val_data["author", "has_work", "work"].edge_label_index
edge_label_val = val_data["author", "has_work", "work"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[100, 50],
    edge_label_index=(("author", "has_work", "work"), edge_label_index_val),
    edge_label=edge_label_val,
    batch_size= 2048 ,
    shuffle=False,
)

# Define the test seed edges:
edge_label_index_test = test_data["author", "has_work", "work"].edge_label_index
edge_label_test = test_data["author", "has_work", "work"].edge_label
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[100, 50],
    edge_label_index=(("author", "has_work", "work"), edge_label_index_test),
    edge_label=edge_label_test,
    batch_size= 2048 ,
    shuffle=False,
)

#calculate val loss
def validate(model, val_loader):
    model.eval()
    total_loss = total_examples = 0
    with torch.no_grad():
        for sampled_data in val_loader:
            sampled_data.to(device)
            pred = model(sampled_data)
            ground_truth = sampled_data["author", "has_work", "work"].edge_label
            loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
            total_loss += float(loss) * pred.numel()
            total_examples += pred.numel()
    model.train()
    return total_loss / total_examples


#final evaluation on the test data
def evaluate(model, test_loader):
    model.eval()
    preds, ground_truths = [], []
    with torch.no_grad():
        for sampled_data in tqdm.tqdm(test_loader):
            sampled_data.to(device)
            preds.append(model(sampled_data))
            ground_truths.append(sampled_data["author", "has_work", "work"].edge_label)
    
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()

    auc = roc_auc_score(ground_truth, pred)
    ap = average_precision_score(ground_truth, pred)
    re = recall_score(ground_truth, pred > 0)
    pre = precision_score(ground_truth, pred > 0)
    acc = accuracy_score(ground_truth, pred > 0)
    f1 = f1_score(ground_truth, pred > 0)
    
    return auc, ap, re, pre, acc, f1


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_edge_features=1):
        super().__init__()
        self.conv1 = GATConv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=num_edge_features)
        self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops=False, edge_dim=num_edge_features)
    
    def forward(self, x, edge_index, edge_attr) -> Tensor:
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = self.conv2(x, edge_index, edge_attr)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_author: Tensor, x_work: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_author = x_author[edge_label_index[0]]
        edge_feat_work = x_work[edge_label_index[1]]
        return (edge_feat_author * edge_feat_work).sum(dim=-1)
    

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.work_lin = torch.nn.Linear(264, hidden_channels)
        self.author_lin = torch.nn.Linear(132, hidden_channels)
        self.source_lin = torch.nn.Linear(139, hidden_channels)
        self.publisher_lin = torch.nn.Linear(134, hidden_channels)
        self.concept_lin = torch.nn.Linear(146, hidden_channels)
        self.institution_lin = torch.nn.Linear(140, hidden_channels)

        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
        

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "work": self.work_lin(data["work"].x),
          "author": self.author_lin(data["author"].x),
          "source": self.source_lin(data["source"].x),
          "publisher": self.publisher_lin(data["publisher"].x),
          "concept": self.concept_lin(data["concept"].x),
          "institution": self.institution_lin(data["institution"].x),
        } 

        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)
        pred = self.classifier(
            x_dict["author"],
            x_dict["work"],
            data["author", "has_work", "work"].edge_label_index,
        )

        return pred 
        
model = Model(hidden_channels=64)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: '{device}'")
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Early Stopping
patience = 1
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(1, 100):
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data)
        ground_truth = sampled_data["author", "has_work", "work"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward(retain_graph=True)
        optimizer.step()
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()

    print(f"Epoch: {epoch:03d}, Training Loss: {total_loss / total_examples:.4f}")
    
    #calcuate val loss
    val_loss = validate(model, val_loader)
    print(f"Epoch: {epoch:03d}, Validation Loss: {val_loss:.4f}")

    #check early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement == patience:
        print(f"Early stopping after {epoch} Epochs.")
        print("Evaluation on Test Data:")
        auc, ap, re, pre, acc, f1_score = evaluate(model, test_loader)
        print(f"Test AUC: {auc:.4f}")
        print(f"Test AP: {ap:.4f}")
        print(f"Test Recall: {re:.4f}")
        print(f"Test Precision: {pre:.4f}")
        print(f"Test Accuracy: {acc:.4f}")
        print(f"Test F1: {f1_score:.4f}")
        break

model_parameters = count_parameters(model)
print(f'The model has {model_parameters:,} trainable parameters.')

sys.stdout = orig_stdout
f.close()