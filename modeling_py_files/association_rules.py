from pyECLAT import ECLAT
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd
from IPython.display import display

def association_rules_preproc(data_w_clusters:pd.DataFrame, encoded_purchases:pd.DataFrame) -> dict:
    '''
    Function that creates a dictionary of cluster information to apply association rules to each cluster

    Inputs:
    - data_w_clusters (pd.Dataframe): Dataframe indexed by customer_id that includes the associated cluster
        We assume the cluster column is 'cluster_solution'
    - encoded_purchases (pd.Dataframe): Dataframe indexed by invoice_id that describes the customers'
    purchases, encoded with transaction encoding

    Outputs:
    - cluster_dfs (dict): Dict of cluster_label | Purchases filtered for that cluster
    '''
    filtered_data = data_w_clusters[data_w_clusters['customer_id'].isin(encoded_purchases['customer_id'])]
    clusters = filtered_data['cluster_solution'].unique()

    encoded_purchases_with_cluster = encoded_purchases.reset_index().merge(filtered_data, on='customer_id', how='left')
    encoded_purchases_with_cluster.drop('customer_id', axis=1, inplace=True)
    encoded_purchases_with_cluster.set_index('invoice_id', inplace=True)

    clustered_baskets = dict()
    for cluster in clusters:
      clustered_baskets[cluster] = encoded_purchases_with_cluster[encoded_purchases_with_cluster['cluster_solution'] == cluster]

    for cluster in clusters:
      clustered_baskets[cluster] = encoded_purchases_with_cluster[encoded_purchases_with_cluster['cluster_solution'] == cluster]
      clustered_baskets[cluster].drop(['cluster_solution'], axis=1, inplace=True)


    return clustered_baskets

def association_rules_apriori(purchases_dict:dict, index:int) -> pd.DataFrame:
    '''
    Function that applies Apriori association rules assuming min_support=0.05

    Inputs:
    purchases_dict (dict): Dictionary of purchases per cluster
      Created by association_rules_preproc()
    index (int): Key to look for in the dict
    
    Outputs:
    Prints the frequent items and rules created of that cluster
    Returns the dataframe of rules
    '''
    frequent_itemsets_grocery = apriori(purchases_dict[index], min_support=0.05, use_colnames=True)

    rules_grocery = association_rules(frequent_itemsets_grocery,
                                    metric="confidence",
                                    min_threshold=0.2)

    # Filter rules where 'lsupport' > 0.1
    rules_grocery_lift_filtered = rules_grocery[rules_grocery['support'] > 0.1]

    print(f'Cluster {index}:')
    display(rules_grocery_lift_filtered.sort_values(by='lift', ascending=False).head(20))
