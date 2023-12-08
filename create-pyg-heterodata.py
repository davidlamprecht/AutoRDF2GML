#Example script for loading the output csv files from AutoRDF2GML into a PyTorch Geometric HeteroData object
#The script contains the output csv files from AutoRDF2GML for the SemOpenAlex-SemanticWeb dataset 

from torch_geometric.data import HeteroData
import pandas as pd
import numpy as np
from torch import Tensor

#Load the heterogenous graph dataset created byAutoRDF2GML into a PyTorch Geometric HeteroData object


#load node csv files
work_csv = "soa-sw/idx/nodes-transe/work_transe.csv"
author_csv = "soa-sw/idx/nodes-transe/author_transe.csv"
publisher_csv = "soa-sw/idx/nodes-transe/publisher_transe.csv"
source_csv = "soa-sw/idx/nodes-transe/source_transe.csv"
concept_csv = "soa-sw/idx/nodes-transe/concept_transe.csv"
institution_csv = "soa-sw/idx/nodes-transe/institution_transe.csv"

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


#load edge csv files
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

work_tensor = torch.tensor(work_df.values, dtype=torch.float)
author_tensor = torch.tensor(author_df.values, dtype=torch.float)
publisher_tensor = torch.tensor(publisher_df.values, dtype=torch.float)
source_tensor = torch.tensor(source_df.values, dtype=torch.float)
concept_tensor = torch.tensor(concept_df.values, dtype=torch.float)
institution_tensor = torch.tensor(institution_df.values, dtype=torch.float)

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

#convert to undirected graph
data = T.ToUndirected()(data)

print(data)
