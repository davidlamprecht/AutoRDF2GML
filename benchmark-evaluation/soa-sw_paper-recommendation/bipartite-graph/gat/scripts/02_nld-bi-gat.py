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
f = open('eval-soa-sw-aw-bipartite/gat/results/02_eval-gat-bi-nld.txt', 'w')
sys.stdout = f

seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)

#load data
work_csv = "soa-sw/idx/nodes-nld/work_nld.csv"
author_csv = "soa-sw/idx/nodes-literals/author_literals.csv"

work_df = pd.read_csv(work_csv, header=None)
author_df = pd.read_csv(author_csv, header=None)

work_df = work_df.astype(float)
author_df = author_df.astype(float)

has_author_csv = "soa-sw/idx/edges/work_author.csv"

has_author_df = pd.read_csv(has_author_csv, header=None)


work_tensor = torch.tensor(work_df.values, dtype=torch.float)
author_tensor = torch.tensor(author_df.values, dtype=torch.float)

has_author_src = torch.tensor(has_author_df.iloc[:, 0].values, dtype=torch.long)
has_author_dst = torch.tensor(has_author_df.iloc[:, 1].values, dtype=torch.long)

data = HeteroData()

data['work'].node_id = torch.arange(len(work_tensor))
data['author'].node_id = torch.arange(len(author_tensor))


#### One-hot encoding
# node types author
num_author_nodes = len(author_df)
one_hot_author = torch.eye(num_author_nodes)
data['author'].x = one_hot_author



# random Projektion
def random_projection(data, original_dim, target_dim):
    # create random matrix with normal distribution
    R = torch.randn((target_dim, original_dim))
    
    # multiply the data with R
    projected_data = torch.mm(data, R.t())
    
    return projected_data

# dimension
target_dim = 64 

#nld work
data['work'].x = work_tensor

#one-hot-encoding other node types
data['author'].x = random_projection(data['author'].x, len(author_df), target_dim)

### end one-hot-encoding node feat


#changed edge direction
data['author', 'has_work', 'work'].edge_index = torch.stack([has_author_dst, has_author_src], dim=0)



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
    def __init__(self, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(hidden_channels, hidden_channels, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class Classifier(torch.nn.Module):
    def forward(self, x_author: Tensor, x_work: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_author = x_author[edge_label_index[0]]
        edge_feat_work = x_work[edge_label_index[1]]
        return (edge_feat_author * edge_feat_work).sum(dim=-1)
    

class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.work_lin = torch.nn.Linear(128, hidden_channels)
        self.author_lin = torch.nn.Linear(64, hidden_channels) 


        self.gnn = GNN(hidden_channels)
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())
        self.classifier = Classifier()
        

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "work": self.work_lin(data["work"].x),
          "author": self.author_lin(data["author"].x),

        } 

        x_dict = self.gnn(x_dict, data.edge_index_dict)
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