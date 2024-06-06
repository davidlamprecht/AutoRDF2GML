from torch_geometric import seed_everything
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
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
f = open('eval-lpwc-dt/hgt/results/03_eval-hgt-literals.txt', 'w')
sys.stdout = f

seed_everything(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


#load data
paper_csv = "lpwc/idx/nodes-literals/paper_literals.csv"
method_csv = "lpwc/idx/nodes-literals/method_literals.csv"
dataset_csv = "lpwc/idx/nodes-literals/dataset_literals.csv"
task_csv = "lpwc/idx/nodes-literals/task_literals.csv"

paper_df = pd.read_csv(paper_csv, header=None)
method_df = pd.read_csv(method_csv, header=None)
dataset_df = pd.read_csv(dataset_csv, header=None)
task_df = pd.read_csv(task_csv, header=None)

paper_df = paper_df.astype(float)
method_df = method_df.astype(float)
dataset_df = dataset_df.astype(float)
task_df = task_df.astype(float)

paper_tensor = torch.tensor(paper_df.values, dtype=torch.float)
method_tensor = torch.tensor(method_df.values, dtype=torch.float)
dataset_tensor = torch.tensor(dataset_df.values, dtype=torch.float)
task_tensor = torch.tensor(task_df.values, dtype=torch.float)


#edges

dataset_paper_csv = "lpwc/idx/edges/dataset_paper.csv"
dataset_task_csv = "lpwc/idx/edges/dataset_task.csv"
method_paper_csv = "lpwc/idx/edges/method_paper.csv"
paper_dataset_csv = "lpwc/idx/edges/paper_dataset.csv"
paper_method_csv = "lpwc/idx/edges/paper_method.csv"
paper_task_csv = "lpwc/idx/edges/paper_task.csv"

dataset_paper_df = pd.read_csv(dataset_paper_csv, header=None)
dataset_task_df = pd.read_csv(dataset_task_csv, header=None)
method_paper_df = pd.read_csv(method_paper_csv, header=None)
paper_dataset_df = pd.read_csv(paper_dataset_csv, header=None)
paper_method_df = pd.read_csv(paper_method_csv, header=None)
paper_task_df = pd.read_csv(paper_task_csv, header=None)



dataset_paper_src = torch.tensor(dataset_paper_df.iloc[:, 0].values, dtype=torch.long)
dataset_paper_dst = torch.tensor(dataset_paper_df.iloc[:, 1].values, dtype=torch.long)
dataset_task_src = torch.tensor(dataset_task_df.iloc[:, 0].values, dtype=torch.long)
dataset_task_dst = torch.tensor(dataset_task_df.iloc[:, 1].values, dtype=torch.long)
method_paper_src = torch.tensor(method_paper_df.iloc[:, 0].values, dtype=torch.long)
method_paper_dst = torch.tensor(method_paper_df.iloc[:, 1].values, dtype=torch.long)
paper_dataset_src = torch.tensor(paper_dataset_df.iloc[:, 0].values, dtype=torch.long)
paper_dataset_dst = torch.tensor(paper_dataset_df.iloc[:, 1].values, dtype=torch.long)
paper_method_src = torch.tensor(paper_method_df.iloc[:, 0].values, dtype=torch.long)
paper_method_dst = torch.tensor(paper_method_df.iloc[:, 1].values, dtype=torch.long)
paper_task_src = torch.tensor(paper_task_df.iloc[:, 0].values, dtype=torch.long)
paper_task_dst = torch.tensor(paper_task_df.iloc[:, 1].values, dtype=torch.long)


data = HeteroData()

data['paper'].node_id = torch.arange(len(paper_tensor))
data['method'].node_id = torch.arange(len(method_tensor))
data['dataset'].node_id = torch.arange(len(dataset_tensor))
data['task'].node_id = torch.arange(len(task_tensor))

data['paper'].x = paper_tensor
data['method'].x = method_tensor
data['dataset'].x = dataset_tensor
data['task'].x = task_tensor

data['paper', 'has_method', 'method'].edge_index = torch.stack([paper_method_src, paper_method_dst], dim=0)
data['paper', 'has_dataset', 'dataset'].edge_index = torch.stack([paper_dataset_src, paper_dataset_dst], dim=0)
data['paper', 'has_task', 'task'].edge_index = torch.stack([paper_task_src, paper_task_dst], dim=0)
data['dataset', 'has_paper', 'paper'].edge_index = torch.stack([dataset_paper_src, dataset_paper_dst], dim=0)
data['dataset', 'has_task', 'task'].edge_index = torch.stack([dataset_task_src, dataset_task_dst], dim=0)
data['method', 'has_paper', 'paper'].edge_index = torch.stack([method_paper_src, method_paper_dst], dim=0)


#finished loading data

data = T.ToUndirected()(data)

#train, val, test split
transform = T.RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=False,
    edge_types=[('dataset', 'has_task', 'task')],
    rev_edge_types=[('task', 'rev_has_task', 'dataset')], 
)
train_data, val_data, test_data = transform(data)


# Define the train seed edges:
edge_label_index_train = train_data["dataset", "has_task", "task"].edge_label_index
edge_label_train = train_data["dataset", "has_task", "task"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[1000, 500],
    neg_sampling_ratio=1.0,
    edge_label_index=(("dataset", "has_task", "task"), edge_label_index_train),
    edge_label=edge_label_train,
    batch_size= 1024 ,
    shuffle=True,
)

# Define the validation seed edges:
edge_label_index_val = val_data["dataset", "has_task", "task"].edge_label_index
edge_label_val = val_data["dataset", "has_task", "task"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[1000, 500],
    edge_label_index=(("dataset", "has_task", "task"), edge_label_index_val),
    edge_label=edge_label_val,
    batch_size= 1024 ,
    shuffle=False,
)

# Define the test seed edges:
edge_label_index_test = test_data["dataset", "has_task", "task"].edge_label_index
edge_label_test = test_data["dataset", "has_task", "task"].edge_label
test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[1000, 500],
    edge_label_index=(("dataset", "has_task", "task"), edge_label_index_test),
    edge_label=edge_label_test,
    batch_size= 1024 ,
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
            ground_truth = sampled_data["dataset", "has_task", "task"].edge_label
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
            ground_truths.append(sampled_data["dataset", "has_task", "task"].edge_label)
    
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
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return x_dict


class Classifier(torch.nn.Module):
    def forward(self, x_dataset: Tensor, x_task: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_dataset = x_dataset[edge_label_index[0]]
        edge_feat_task = x_task[edge_label_index[1]]
        return (edge_feat_dataset * edge_feat_task).sum(dim=-1)
    

class Model(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.paper_lin = torch.nn.Linear(129, out_channels)
        self.method_lin = torch.nn.Linear(130, out_channels)
        self.dataset_lin = torch.nn.Linear(132, out_channels)
        self.task_lin = torch.nn.Linear(128, out_channels)

        self.gnn = GNN(hidden_channels=64, out_channels= out_channels, num_heads=8, num_layers=2)
        self.classifier = Classifier()
        

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "paper": self.paper_lin(data["paper"].x),
          "method": self.method_lin(data["method"].x),
          "dataset": self.dataset_lin(data["dataset"].x),
          "task": self.task_lin(data["task"].x),

        } 

        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["dataset"],
            x_dict["task"],
            data['dataset', 'has_task', 'task'].edge_label_index,
        )

        return pred 
        
model = Model(out_channels=64)


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
        ground_truth = sampled_data["dataset", "has_task", "task"].edge_label
        loss = F.binary_cross_entropy_with_logits(pred, ground_truth)
        loss.backward()
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