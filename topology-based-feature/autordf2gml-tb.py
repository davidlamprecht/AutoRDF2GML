import rdflib
from rdflib import URIRef, BNode, Literal
import numpy as np
import configparser
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Dataset, download_url, Data
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric import seed_everything
# from torch_geometric.nn import TransE
from torch_geometric.nn import ComplEx, DistMult, TransE
import torch
import torch.optim as optim
import csv
import os

import argparse, time
from tqdm import tqdm

def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--config_path", type=str, default='use-case-aifb/config-aifb.ini')
  # parser.add_argument("--config", type=str, default='config.ini')
  return parser

def folder_check(mpath):
  if os.path.isdir(mpath): 
    print (f'## Path exists: {mpath}')
  else: 
    os.makedirs(mpath, exist_ok=True)
    print (f'## Path {mpath} created!')

model_map = {
    'transe': TransE,
    'complex': ComplEx,
    'distmult': DistMult,
    #'rotate': RotatE,
}

args = _get_parser().parse_args()

#Define the path to the config file
config = configparser.ConfigParser()
print(f"## AutoRDF2GML (topology-based) ##")
start_time = time.time()
print(f"## {start_time=}")
print(f"## Loading the config file: {args.config_path}")
config.read(args.config_path)
# config.read('use-case-aifb/config-cb.ini')

##########################################################################################
#
# START AutoRDF2GML Topology-based (TB) Node Features Version
#
##########################################################################################

#Parse the config
file_path = config.get('InputPath', 'input_path')
save_path_numeric_graph = config.get('SavePath', 'save_path_numeric_graph')
save_path_mapping = config.get('SavePath', 'save_path_mapping')
kge_model = config.get('MODEL', 'kge_model')

print (f'## Configs: input:{file_path} / output:{save_path_mapping} {save_path_numeric_graph} / {kge_model=}')

folder_check(save_path_numeric_graph)
folder_check(save_path_mapping)

graph = rdflib.Graph()
print(f"## Loading the RDF dump from: {file_path=}...")
graph.parse(file_path, format="nt")
print(f"## RDF dump file loaded. The RDF graph contains {len(graph)} triples.")

##########################################################################################
#
# START TOPOLGY-BASED NODE FEATURES CREATION
#
##########################################################################################

class_list_str = config.get('EmbeddingClasses', 'class_list')
class_list = [URIRef(uri.strip()) for uri in class_list_str.split(', ')]

pred_list_str = config.get('EmbeddingPredicates', 'pred_list')
pred_list = [URIRef(uri.strip()) for uri in pred_list_str.split(', ')]

print(f"## Transformation started! Automatic features extraction..")

#RDF Data Preprocessing

#Initialisation of the counters and dictionaries
entity_counter = 0
relation_counter = 0
entity_dict = {}
relation_dict = {}

triples = []

#Iteration over the triples in the graph
for s, p, o in graph:
    if p == URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"):
        if o not in class_list:
            continue
    else:
        o_classes = [obj for obj in graph.objects(o, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))]
        s_classes = [obj for obj in graph.objects(s, URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"))]

        if type(o) is URIRef and not any(oc in class_list for oc in o_classes):
            continue
        if type(s) is URIRef and not any(sc in class_list for sc in s_classes):
            continue

    if type(o) is not URIRef or p not in pred_list:
        continue

    if s not in entity_dict and s not in class_list:
        entity_dict[s] = entity_counter
        entity_counter += 1

    if o not in entity_dict and o not in class_list:
        entity_dict[o] = entity_counter
        entity_counter += 1

    if p not in relation_dict:
        relation_dict[p] = relation_counter
        relation_counter += 1

    #Test if subject, predicate and object are in the mapping before adding the triple
    if s in entity_dict and p in relation_dict and o in entity_dict:
        triples.append((entity_dict[s], relation_dict[p], entity_dict[o]))


#Initialisation of three empty lists for the subjects, predicates and objects
first_numbers = []
second_numbers = []
third_numbers = []

#Iterate over the triples in the list and append the subjects, predicates and objects to the lists
for s, p, o in triples:
    first_numbers.append(s)
    second_numbers.append(p)
    third_numbers.append(o)

first_tensor = torch.tensor(first_numbers)
second_tensor = torch.tensor(second_numbers)
third_tensor = torch.tensor(third_numbers)

combined_tensor = torch.stack((first_tensor, third_tensor))


data = Data(edge_index=combined_tensor,
            edge_type=second_tensor,
            num_nodes=combined_tensor.max().item() + 1,
            ) 

#optional seed for reproducibility
seed_everything(42)

#we use 100% of the data for training
transform = RandomLinkSplit(
    num_val=0.0,
    num_test=0.0,
)
train_data, val_data, test_data = transform(data)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Define the model. Default model is TransE with 128 hidden channels
model_arg_map = {'rotate': {'margin': 9.0}}
model = model_map[kge_model](
    num_nodes=train_data.num_nodes,
    num_relations=train_data.num_edge_types,
    hidden_channels=128,
    **model_arg_map.get(kge_model, {}),
).to(device)

# model = TransE(
#     num_nodes=train_data.num_nodes,
#     num_relations=train_data.num_edge_types,
#     hidden_channels=128,
# ).to(device)

#Default Batch size is 2000
loader = model.loader(
    head_index=train_data.edge_index[0].to(device),
    rel_type=train_data.edge_type.to(device),
    tail_index=train_data.edge_index[1].to(device),
    batch_size=2000,
    shuffle=True,
)

#Default optimizer is Adam with learning rate 0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train():
    model.train()
    total_loss = total_examples = 0
    for head_index, rel_type, tail_index in loader:
        optimizer.zero_grad()
        loss = model.loss(head_index, rel_type, tail_index)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * head_index.numel()
        total_examples += head_index.numel()
    return total_loss / total_examples


@torch.no_grad()
def test(data):
    model.eval()
    head_index = val_data.edge_index[0].to(device)
    rel_type = val_data.edge_type.to(device)
    tail_index = val_data.edge_index[1].to(device)
    return model.test(
        head_index=head_index,
        rel_type=rel_type,
        tail_index=tail_index,
        batch_size=20000,
        k=10,
    )

#Default number of epochs is 900
print(f"## Training the KG embedding...")
for epoch in tqdm(range(1, 901), desc=f'Training'):
    loss = train()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

entity_embeddings = model.node_emb.weight.cpu().detach().numpy()
relation_embeddings = model.rel_emb.weight.cpu().detach().numpy()

##########################################################################################
#
# END TOPOLGY-BASED NODE FEATURES CREATION
#
##########################################################################################

print(f"## Features creation done! Continue.. ")

#get the defined names for the classes and edges from the config file
class_names = config.get('Nodes', 'classes').split(', ')
edge_names_simple = config.get('SimpleEdges', 'edge_names').split(', ')
try:
    edge_names_n_hop = config.get('N-HopEdges', 'edge_names').split(', ')
except:
    edge_names_n_hop = None
    print (f'No N-HopEdges found! Continue..')

#create dictionaries for classes
class_dict = {class_name: [rdflib.URIRef(uri.strip()) for uri in config.get('Nodes', class_name).split(',')] for class_name in class_names}

#create dictionaries for simple edges
simple_edge_dict = {}
for edge_name in edge_names_simple:
    start_node_name = config.get('SimpleEdges', f'{edge_name}_start_node')
    start_node = class_dict[start_node_name] # Keine zusätzlichen eckigen Klammern
    properties = config.get('SimpleEdges', f'{edge_name}_properties').split(', ')
    end_node_name = config.get('SimpleEdges', f'{edge_name}_end_node')
    end_node = class_dict[end_node_name] # Keine zusätzlichen eckigen Klammern
    simple_edge_dict[edge_name] = [start_node, properties, end_node]

#create dictonaries for n-hop edges

if edge_names_n_hop:
    n_hop_edge_dict = {}
    for edge_name in edge_names_n_hop:
        start_node = class_dict[config.get('N-HopEdges', edge_name + '_start_node')]
        end_node = class_dict[config.get('N-HopEdges', edge_name + '_end_node')]
        n_hop_edge = [start_node]
        hop_index = 1
        while True:
            hop_key = edge_name + '_hop' + str(hop_index) + '_properties'
            if config.has_option('N-HopEdges', hop_key):
                properties = config.get('N-HopEdges', hop_key).split(', ')
                n_hop_edge.append(properties)  
                hop_index += 1
            else:
                break
        n_hop_edge.append(end_node) # Füge den Endknoten am Ende hinzu
        n_hop_edge_dict[edge_name] = n_hop_edge


    
#creates lists for the uris
uri_lists = {}
for class_name in class_names:
    uri_list_name = f"uri_list_{class_name}"
    globals()[uri_list_name] = []
    uri_lists[uri_list_name] = globals()[uri_list_name]


nodes_data_df = {}
for class_name in class_names:
    nodes_data_df[f'df_{class_name}'] = pd.DataFrame(columns=["subject", "predicate", "object"])

nodes_data_pivoted_df = {}
for class_name in class_names:
    nodes_data_pivoted_df[f'pivoted_df_{class_name}'] = pd.DataFrame()

for uri_list, node_class in zip(uri_lists, class_dict.values()):
    entity_list = uri_lists[uri_list]

    query = """
        SELECT DISTINCT ?entity
        WHERE {
            ?entity rdf:type ?class .
        }
    """

    for class_uri in node_class:
        query_with_class = query.replace("?class", f"<{class_uri}>")
        for row in graph.query(query_with_class):
            entity_uri = row[0]
            if entity_uri not in entity_list:
                entity_list.append(entity_uri)

output_path = save_path_numeric_graph
#save the topological node features
# print (f'{uri_lists=}')

for uri_list in uri_lists:
    entity_list = uri_lists[uri_list]

    rows_for_df = []
    
    file_path = os.path.join(output_path, f'{uri_list}.csv')
    
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        embedding_size = len(entity_embeddings[next(iter(entity_dict.values()))])
        columns = ['subject'] + [f'feature_{i+1}' for i in range(embedding_size)]
        writer.writerow(columns)

        for entity in entity_list:
            try:
                entity_id = entity_dict[entity] 
                embedding = entity_embeddings[entity_id]
                row = [entity] + embedding.tolist()
                writer.writerow(row)
                rows_for_df.append(row)
            except:
                print (f'Entity not in triples: {entity=}')
                continue

    df = pd.DataFrame(rows_for_df, columns=columns)
    nodes_data_pivoted_df[f'pivoted_df_{uri_list}'] = df

# print (f'{nodes_data_pivoted_df=}')
##########################################################################################
#
# START EDGE LIST CONSTRUCTION
#
##########################################################################################

print(f"## Edge list construction...")

#Binary edges (refed to as simple edges)
simple_edge_lists = {}
for var_name in simple_edge_dict.keys():
    edge_list_name = f"edge_list_{var_name}"
    globals()[edge_list_name] = []
    simple_edge_lists[edge_list_name] = globals()[edge_list_name]

for edges, edge_list in zip(simple_edge_dict.values(), simple_edge_lists.values()):
    subject_value, predicte_value, object_value = edges
    for a in subject_value:
        for b in object_value:
            for p in predicte_value:
                query = """
                    SELECT DISTINCT ?a ?c
                    WHERE {
                        ?a rdf:type ?class_a .
                        ?c rdf:type ?class_b .
                        ?a ?b ?c .
                        }
                    """
            
                query_replace = query.replace("?class_a", f"<{a}>").replace("?class_b", f"<{b}>").replace("?b", f"<{p}>")
                for row in graph.query(query_replace):
                    edge_list.append(row)
                    

##### binary edges done ######

#n-hop edges
if edge_names_n_hop:
    n_hop_edge_lists = {}
    for var_name in n_hop_edge_dict.keys():
        edge_list_name = f"edge_list_{var_name}"
        globals()[edge_list_name] = []
        n_hop_edge_lists[edge_list_name] = globals()[edge_list_name]


    def nested_loops(list_of_lists, result_list, class_a, class_x):
        list_of_lists = list_of_lists[1:-1]

        def _nested_loops_recursion(lists, current_combination):
            if not lists:
                query_a = create_sparql_query(current_combination, class_a, class_x)
    
                for row in graph.query(query_a):
                    result_list.append(row)
                
                return

            for item in lists[0]:
                _nested_loops_recursion(lists[1:], current_combination + [item])

        _nested_loops_recursion(list_of_lists, [])


    def create_sparql_query(current_combination, class_a, class_x):
        triples = ""
        prev_var = "?a"
        for i, prop in enumerate(current_combination):
            var = f"?c{i+1}" if i < len(current_combination) - 1 else "?x"
            triples += f"{prev_var} <{prop}> {var} .\n"
            prev_var = var

        query_a = f"""
            SELECT DISTINCT ?a ?x
            WHERE {{
                ?a rdf:type <{class_a}> .
                ?x rdf:type <{class_x}> .
                {triples}
            }}
        """
        return query_a


    for nhop_edge, nhop_list in zip(n_hop_edge_dict.values(), n_hop_edge_lists.values()):
        class_a_list = nhop_edge[0]
        class_x_list = nhop_edge[-1]
        
        for class_a in class_a_list:
            for class_x in class_x_list:
                nested_loops(nhop_edge, nhop_list, class_a, class_x)

##### n-hop edges done ######


##########################################################################################
#
# END EDGE LIST CONSTRUCTION
#
##########################################################################################

print(f"## Features creation done! Continue.. ")

print(f"## Mapping.. ")

#map the uris to idx
mapping_df = {}
for var_name in class_names:
    mapping_df[f'mapping_df_{var_name}'] = pd.DataFrame()

def read_mapping(mapping_df):
    mapping = {}
    for _, row in mapping_df.iterrows():
        mapping[row[1]] = row[0]
    return mapping

# print (nodes_data_pivoted_df.items())

for data_pivoted_df, data_mapping_df in zip(nodes_data_pivoted_df.items(), mapping_df.items()):
    try:
        # print (f'{data_pivoted_df=}, {data_mapping_df=}')
        key_pivot, value_pivot = data_pivoted_df
        key_mapping, value_mapping = data_mapping_df
        
        unique_user_id = value_pivot["subject"].unique()
        unique_user_id = pd.DataFrame(data={
        'userId': unique_user_id,
        'mappedID': pd.RangeIndex(len(unique_user_id)),
        })
    
        
        mapping_df[key_mapping] = read_mapping(unique_user_id)
        for key, value in mapping_df[key_mapping].items():
            if isinstance(value, URIRef):
                mapping_df[key_mapping][key] = str(value)
    except:
        continue
            


for key, value in nodes_data_pivoted_df.items():
    try:
        value_copy = value.copy()
        value_copy.drop(['subject'], axis=1, inplace=True)
        filename = key + ".csv"
        file_path = os.path.join(save_path_numeric_graph, filename)
        value_copy.to_csv(file_path, index=False, header=False)
    except:
        continue


for key, value in nodes_data_pivoted_df.items():
    try:
        value_copy = value[['subject']].copy()
        value_copy['mapping'] = range(len(value_copy))
        filename = key + ".csv"
        file_path = os.path.join(save_path_mapping, filename)
        value_copy.to_csv(file_path, index=False)
    except:
        continue


def invert_mapping(mapping):
    return {v: k for k, v in mapping.items()}

print(f"## Saving.. ")

#save binary edges (simple edges)
for key, value in simple_edge_lists.items():
    df = pd.DataFrame(value) 
    
    for key_mapping, value_mapping in mapping_df.items():
        inverted_mapping = invert_mapping(value_mapping)
        df = df.astype(str)
        df = df.replace(inverted_mapping) 
    

    filename = key + ".csv" 
    file_path = os.path.join(save_path_numeric_graph, filename)
    
    df.to_csv(file_path, index=False, header=False)



#save n-hop edges
if edge_names_n_hop:
    for key, value in n_hop_edge_lists.items():
        df = pd.DataFrame(value) 
        
        for key_mapping, value_mapping in mapping_df.items():
            inverted_mapping = invert_mapping(value_mapping)
            df = df.astype(str)
            df = df.replace(inverted_mapping) 
        
        filename = key + ".csv" 
        file_path = os.path.join(save_path_numeric_graph, filename) 
        
        df.to_csv(file_path, index=False, header=False)


print(f"## Result saved at: {save_path_mapping=} {save_path_numeric_graph}")

print(f"## Finished creating the graph dataset!")

######## Automatic graph creation done ########
print("--- %.2f seconds ---" % (time.time() - start_time))
print(f"## AutoRDF2GML (topology-based): END!")
