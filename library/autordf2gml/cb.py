##########################################################################################
#
# START AutoRDF2GML Content-based (CB) Node Features Version
#
##########################################################################################
import sys
import rdflib
from rdflib import Literal, URIRef
import pandas as pd
import numpy as np
import configparser
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import time
import torch
from transformers import AutoTokenizer, AutoModel, logging
from tqdm import tqdm
from .utils import *
logging.set_verbosity_error()

def content_feature(config):
    print(f"###### AutoRDF2GML (content-based): START! ######")
    start_time = time.time()
    print(f"## {start_time=}")
    config = load_config(config)
    #Parse the config
    file_path = config.get('InputPath', 'input_path')
    save_path_numeric_graph = config.get('SavePath', 'save_path_numeric_graph')
    save_path_mapping = config.get('SavePath', 'save_path_mapping')
    nld_class = config.get('NLD', 'nld_class')
    pivoted_df_nld = f"pivoted_df_{nld_class}"
    embedding_model = config.get('EMBEDDING', 'embedding_model')

    folder_check(save_path_numeric_graph)
    folder_check(save_path_mapping)

    print (f'## Configs: input:{file_path} / {nld_class=} / {embedding_model=} / output:{save_path_mapping=} {save_path_numeric_graph}')

    #get the defined names for the classes and edges from the config file
    class_names = config.get('Nodes', 'classes').split(', ')
    edge_names_simple = config.get('SimpleEdges', 'edge_names').split(', ')
    edge_names_n_aray = config.get('N-ArayEdges', 'edge_names').split(', ')
    edge_names_n_hop = config.get('N-HopEdges', 'edge_names').split(', ')

    graph = rdflib.Graph()
    fileformat = file_path.split('.')[-1]
    print(f"## Loading the RDF dump from: {file_path=}, format: {fileformat}")
    try:
      # graph.parse(file_path, format="nt")
      graph.parse(file_path, format=fileformat)
      print(f"## RDF dump file loaded. The RDF graph contains {len(graph)} triples.")
    except:
      print(f"## Error loading the input: {file_path=} ! Please check your RDF dump file. ")
      sys.exit(1)

    print(f"## Transformation started! Querying the graph...")

    #create dictionaries for classes
    class_dict = {class_name: [rdflib.URIRef(uri.strip()) for uri in config.get('Nodes', class_name).split(',')] for class_name in class_names}

    #create dictionaries for simple edges
    simple_edge_dict = {}
    for edge_name in edge_names_simple:
        start_node_name = config.get('SimpleEdges', f'{edge_name}_start_node')
        start_node = class_dict[start_node_name]
        properties = config.get('SimpleEdges', f'{edge_name}_properties').split(', ')
        end_node_name = config.get('SimpleEdges', f'{edge_name}_end_node')
        end_node = class_dict[end_node_name]
        simple_edge_dict[edge_name] = [start_node, properties, end_node]

    #create dictonaries for n-aray edges
    n_aray_edge_dict = {}
    for edge_name in edge_names_n_aray:
        start_node_name = config.get('N-ArayEdges', f'{edge_name}_start_node')
        start_node = class_dict[start_node_name]
        properties = config.get('N-ArayEdges', f'{edge_name}_properties').split(', ')
        end_node_name = config.get('N-ArayEdges', f'{edge_name}_end_node')
        end_node = class_dict[end_node_name]
        n_aray_edge_dict[edge_name] = [start_node, properties, end_node]

    #create dictonaryies for n-aray feature paths and feature values
    n_aray_feature_path_dict = {}
    n_aray_feature_value_dict = {}
    for edge_name in edge_names_n_aray:
        n_aray_feature_path_dict[edge_name + '_feature_path'] = config.get('N-ArayFeaturePath', edge_name + '_feature_path').split(', ')
        n_aray_feature_value_dict[edge_name + '_feature_value'] = config.get('N-ArayFeatureValue', edge_name + '_feature_value')

    #create dictonaries for n-hop edges
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
        n_hop_edge.append(end_node) 
        n_hop_edge_dict[edge_name] = n_hop_edge



    #creates lists for the entity uris
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



    for node_class, uri_list, node_data_df in zip(class_dict.values(), uri_lists, nodes_data_df.items()):
        key, value = node_data_df
        entity_list = uri_lists[uri_list]

        query = """
            SELECT DISTINCT ?a ?b ?c 
            WHERE {
                ?a rdf:type ?class .
                ?a ?b ?c .
            }
          """
        all_rows = []
        for class_uri in node_class:
            query_with_uri = query.replace("?class", f"<{class_uri}>")
            for row in graph.query(query_with_uri):
                new_row = {
                    "subject": row[0],
                    "predicate": row[1],
                    "object": row[2]
                }
                all_rows.append(new_row)
        
        new_rows_df = pd.DataFrame(all_rows, columns=["subject", "predicate", "object"])
        value = pd.concat([value, new_rows_df], ignore_index=True)
        nodes_data_df[key] = value


    #deletes object properties with URIRef
    for key, value in nodes_data_df.items():
        indices_to_drop = value[value['object'].apply(lambda x: isinstance(x, rdflib.URIRef))].index
        
        value.drop(indices_to_drop, inplace=True)

    def join_with_comma(x):
        return ', '.join(map(str, x))

    for data_df, data_pivoted_df in zip(nodes_data_df.items(), nodes_data_pivoted_df.items()):
        key, value = data_df
        key_pivot, value_pivot = data_pivoted_df
        
        grouped_df = nodes_data_df[key].groupby(['subject', 'predicate'])['object'].apply(join_with_comma).reset_index()

        pivoted_df_temp = grouped_df.pivot(index='subject', columns='predicate', values='object').reset_index()

        nodes_data_pivoted_df[key_pivot] = pd.concat([value_pivot, pivoted_df_temp], axis=1)



    ##########################################################################################
    #
    # START AUTOMATIC FEATURE SELECTION AND TRANSFORMATION
    #
    ##########################################################################################

    print(f"## Automatic feature selection...")

    #Find NLD columns
    print (f'## NLD column: {pivoted_df_nld}')

    text_columns = []
    for col in nodes_data_pivoted_df["pivoted_df_work"].columns:
        sample_values = nodes_data_pivoted_df["pivoted_df_work"][col].head(1000).dropna()

        unique_strings = sample_values.nunique()
        
        avg_spaces = sample_values.apply(lambda x: str(x).count(' ')).mean()

        if unique_strings > 3 and avg_spaces > 3:
            text_columns.append(col)


    cols_to_keep = []
    cols_to_keep.append('subject')
    cols_to_keep.append('string-values')

    for col in text_columns:
        cols_to_keep.append(col)


    #merge NLD columns
    def merge_columns(df, text_columns):
        try:
            df['string-values'] = df[text_columns].apply(lambda row: ' '.join(row.dropna().map(str)), axis=1)

            df = df.drop(columns=text_columns)

        except KeyError:
            pass

        return df


    #embed nld strings with scibert
    def embed_strings(dataframe, column):
        # tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        # model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        model = AutoModel.from_pretrained(embedding_model)
        
        def random_embedding(output_size=128):
            return np.random.randn(output_size)

        def embed_abstract(abstract, max_length=512, output_size=128):
            if pd.isna(abstract) or abstract == "":
                return random_embedding(output_size)

            tokens = tokenizer(abstract, max_length=max_length, truncation=True, padding='max_length', return_tensors='pt')
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state

            reduced_embeddings = torch.mean(embeddings, dim=1)[:, :output_size]
            return reduced_embeddings.numpy().squeeze()
        
        try:
            tqdm.pandas(desc="Embedding Progress")
            dataframe['embeddings'] = dataframe[column].progress_apply(embed_abstract)
        except KeyError:
            return dataframe

        return dataframe



    def convert_to_string(df):
        return df.applymap(lambda x: str(x) if isinstance(x, (Literal, URIRef)) else x)


    #automatic features selection (preprocessing)
    def preprocess_dataframe(df, cols_to_keep):
        total_rows = len(df)
        for col in df.columns:
            if col not in cols_to_keep:
                missing_values = df[col].isnull().sum()
                unique_values = len(df[col].dropna().unique())
                unique_percent = unique_values / len(df[col].dropna())
                missing_percent = missing_values / total_rows
                identical_precentage = unique_values 
                if unique_percent > 0.90 or missing_percent > 0.25 or unique_values == 1:
                    del df[col]
        return df

    #remove highly correlated columns
    def remove_highly_correlated_columns(df, cols_to_keep):
        df_corr = df.copy()
        
        for col in df_corr.columns:
            if col not in cols_to_keep: 
                df_corr[col] = df_corr[col].astype('category').cat.codes

        corr_matrix = df_corr.corr(method='pearson')

        columns_to_drop = []

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.95:
                    col_to_drop = corr_matrix.columns[i] if df_corr[corr_matrix.columns[i]].count() < df_corr[corr_matrix.columns[j]].count() else corr_matrix.columns[j]
                    if col_to_drop not in cols_to_keep: 
                        columns_to_drop.append(col_to_drop)

        df.drop(columns_to_drop, axis=1, inplace=True)
        
        return df


    def one_hot_encode_categorical_columns(df):
        cat_cols = [col for col in df.columns if df[col].dtype == 'object' and 2 <= df[col].nunique() <= 10]
        
        for col in cat_cols:
            if col not in cols_to_keep:
                # One-hot encode with NaN as a separate category
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df = pd.concat([df, dummies], axis=1)

                # Drop the original column after one-hot encoding
                df.drop(columns=[col], inplace=True)
                
        return df


    def label_encode_categorical_columns(df):
        le = LabelEncoder()

        cat_cols = [col for col in df.columns if df[col].dtype == 'object' and 10 < df[col].nunique() <= 100]
        
        for col in cat_cols:
            if col not in cols_to_keep:
                mask = df[col].notna() & (df[col] != '')
                
                # Convert only non-NaN and non-empty values to string and then encode them
                df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
                
                # Replace NaN and empty values with median of the column
                median_value = df[col].astype(np.float64).median()
                df[col].fillna(median_value, inplace=True)
                df.loc[df[col] == '', col] = median_value

        return df


    def convert_datetime_to_unix(df):
        def majority_is_timestamp(col):
            count = 0
            for value in col:
                try:
                    pd.to_datetime(value, format='%Y-%m-%d')
                    count += 1
                except:
                    pass
            return count / len(col) > 0.5

        for col in df.columns:
            if df[col].dtype == 'object' and majority_is_timestamp(df[col]):

                try:
                    # Convert the column to datetime format
                    df[col] = pd.to_datetime(df[col], errors='coerce', format='%Y-%m-%d')
                    
                    # Convert datetime values to timestamp; leave NaN values as they are
                    df[col] = df[col].apply(lambda x: x.timestamp() if not pd.isna(x) else np.nan)
                    
                    # Compute mean timestamp and replace NaN values with it
                    mean_timestamp = df[col].mean()
                    df[col].fillna(mean_timestamp, inplace=True)
                except:
                    pass

        return df



    def delete_uri_columns(df: pd.DataFrame) -> pd.DataFrame:
        for column in df.columns:
            if column not in cols_to_keep:
                if df[column].apply(lambda x: isinstance(x, str) and (x.startswith('https://') or x.startswith('http://'))).any():
                    df = df.drop(column, axis=1)
        return df



    def normalize_large_values(df):
        scaler = MinMaxScaler()

        for column in df.columns:
            try:
                df[column] = pd.to_numeric(df[column], errors='raise')
            except ValueError:
                continue

            max_val = df.loc[:, column].max()
            min_val = df.loc[:, column].min()
            if max_val > 3 or min_val < -3:
                print("Column", column, "has values outside the range of -3 and 3.")
                df.loc[:, column] = scaler.fit_transform(df.loc[:, column].values.reshape(-1, 1))

        return df


    #Apply the automatic feature selection and transformation
    for key, value in nodes_data_pivoted_df.items(): 
        
        value = merge_columns(value, text_columns)
        value = preprocess_dataframe(value, cols_to_keep)
        value = one_hot_encode_categorical_columns(value)
        value = label_encode_categorical_columns(value)
        value = delete_uri_columns(value)    
        value = remove_highly_correlated_columns(value, cols_to_keep)
        value = convert_datetime_to_unix(value)
        value = normalize_large_values(value)
        value = embed_strings(value, 'string-values')

        # Update the value in the original dictionary
        nodes_data_pivoted_df[key] = value
        

    #split the nld embeddings column into sepeate columns
    for key, value in nodes_data_pivoted_df.items():
        if 'embeddings' in value.columns:

            embeddings = pd.DataFrame(value['embeddings'].to_list(), columns=[f'embedding_{i}' for i in range(128)])

            value.drop('embeddings', axis=1, inplace=True)

            value = pd.concat([value.reset_index(drop=True), embeddings.reset_index(drop=True)], axis=1)

            nodes_data_pivoted_df[key] = value


    #Delte the 'string-values' column
    for key, value in nodes_data_pivoted_df.items():
        if 'string-values' in value.columns:
            value.drop('string-values', axis=1, inplace=True)
            nodes_data_pivoted_df[key] = value



    ##########################################################################################
    #
    # END AUTOMATIC FEATURE SELECTION AND TRANSFORMATION
    #
    ##########################################################################################



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

    #N-aray edges with features
    n_aray_edge_lists = {}
    for var_name in n_aray_edge_dict.keys():
        edge_list_name = f"edge_list_{var_name}"
        globals()[edge_list_name] = []
        n_aray_edge_lists[edge_list_name] = globals()[edge_list_name]

    n_aray_edge_feature_lists = {}
    for var_name in n_aray_edge_dict.keys():
        edge_list_name = f"edge_feature_list_{var_name}"
        globals()[edge_list_name] = []
        n_aray_edge_feature_lists[edge_list_name] = globals()[edge_list_name]

    for naray_e, naray_feature_path, naray_feature_value, edge_list, edge_feature_list in zip(n_aray_edge_dict.values(), n_aray_feature_path_dict.values(), n_aray_feature_value_dict.values(), n_aray_edge_lists.values(), n_aray_edge_feature_lists.values()):
        subject_value, predicte_value, object_value = naray_e
        
        for a in subject_value:
            for b in object_value:
                for p in predicte_value:
                    query_a = """
                            SELECT DISTINCT ?a ?c
                            WHERE {
                                ?a rdf:type ?class_a .
                                ?c rdf:type ?class_b .
                                ?a ?b ?c .
                            }
                        """
                    query_replace_a = query_a.replace("?class_a", f"<{a}>").replace("?class_b", f"<{b}>").replace("?b", f"<{p}>")
                    for row in graph.query(query_replace_a):
                        edge_list.append(row)
        
        
        for row in edge_list:
            uri_a = row[0]
            uri_b = row[1]


            query_b = """
            SELECT DISTINCT ?f
            WHERE {
                $uri_a ?b ?c .
                ?c ?d $uri_b .
                ?c ?e ?f .
            }
            """  
            propertie_a = naray_feature_path[0]
            propertie_b = naray_feature_path[1]
            propertie_c = naray_feature_value
            query_replace_b = query_b.replace("$uri_a", f"<{uri_a}>").replace("$uri_b", f"<{uri_b}>").replace("?b", f"<{propertie_a}>").replace("?d", f"<{propertie_b}>").replace("?e", f"<{propertie_c}>")

            result = [row["f"].toPython() for row in graph.query(query_replace_b)]
            edge_feature_list.append(result)
                        
    ##### n-aray edges done ######


    #n-hop edges
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

    print(f"## Saving the result...")
    folder_check(save_path_numeric_graph)
    folder_check(save_path_mapping)

    #map the uris to idx
    mapping_df = {}
    for var_name in class_names:
        mapping_df[f'mapping_df_{var_name}'] = pd.DataFrame()


    def read_mapping(mapping_df):
        mapping = {}
        for _, row in mapping_df.iterrows():
            mapping[row[1]] = row[0]
        return mapping


    for data_pivoted_df, data_mapping_df in zip(nodes_data_pivoted_df.items(), mapping_df.items()):
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
                


    for key, value in nodes_data_pivoted_df.items():
        value_copy = value.copy()
        value_copy.drop(['subject'], axis=1, inplace=True)
        filename = key + ".csv"
        file_path = os.path.join(save_path_numeric_graph, filename)
        value_copy.to_csv(file_path, index=False, header=False)


    for key, value in nodes_data_pivoted_df.items():
        value_copy = value[['subject']].copy()
        value_copy['mapping'] = range(len(value_copy))
        filename = key + ".csv"
        file_path = os.path.join(save_path_mapping, filename)
        value_copy.to_csv(file_path, index=False)


    def invert_mapping(mapping):
        return {v: k for k, v in mapping.items()}


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



    #save n-aray edges
    for key, value in n_aray_edge_lists.items():
        df = pd.DataFrame(value)
        
        for key_mapping, value_mapping in mapping_df.items():
            inverted_mapping = invert_mapping(value_mapping)
            df = df.astype(str)
            df = df.replace(inverted_mapping) 

        filename = key + ".csv" 
        file_path = os.path.join(save_path_numeric_graph, filename) 
        
        df.to_csv(file_path, index=False, header=False)


    #save n-hop edges
    for key, value in n_hop_edge_lists.items():
        df = pd.DataFrame(value) 
        
        for key_mapping, value_mapping in mapping_df.items():
            inverted_mapping = invert_mapping(value_mapping)
            df = df.astype(str)
            df = df.replace(inverted_mapping) 
        
        filename = key + ".csv" 
        file_path = os.path.join(save_path_numeric_graph, filename) 
        
        df.to_csv(file_path, index=False, header=False)



    #Transform edge features
    def edge_label_encode_categorical_columns(df):
        le = LabelEncoder()
        cat_cols = [col for col in df.columns if df[col].dtype == 'object' and 5 < df[col].nunique() <= 100]
        for col in cat_cols:
            if col not in cols_to_keep:
                mask = df[col].notna() & (df[col] != '')
                # Convert only non-NaN and non-empty values to string and then encode them
                df.loc[mask, col] = le.fit_transform(df.loc[mask, col].astype(str))
                # Replace NaN and empty values with median of the column
                median_value = df[col].astype(np.float64).median()
                df[col].fillna(median_value, inplace=True)
                df.loc[df[col] == '', col] = median_value

        return df


    def edge_one_hot_encode_categorical_columns(df):
        cat_cols = [col for col in df.columns if df[col].dtype == 'object' and 1 <= df[col].nunique() <= 5]

        for col in cat_cols:
            if col not in cols_to_keep:
                # One-hot encode with NaN as a separate category
                dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
                df = pd.concat([df, dummies], axis=1)
                # Drop the original column after one-hot encoding
                df.drop(columns=[col], inplace=True)

        return df


    for key, value in n_aray_edge_feature_lists.items():
        
        df = pd.DataFrame({'Value': pd.Series([item[0] for item in value])})
        df = edge_one_hot_encode_categorical_columns(df)
        df = edge_label_encode_categorical_columns(df)
        filename = key + ".csv" 
        file_path = os.path.join(save_path_numeric_graph, filename)
        
        df.to_csv(file_path, index=False, header=False)



    print(f"## Result saved at: {save_path_mapping=} {save_path_numeric_graph}")

    print(f"## Finished creating the graph dataset!")

    ######## Automatic graph creation done ########

    print("--- %.2f seconds ---" % (time.time() - start_time))
    print(f"###### AutoRDF2GML (content-based): END! ######")