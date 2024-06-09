from pyECLAT import ECLAT
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

def association_rules_preproc(data_w_clusters:pd.DataFrame, encoded_purchases:pd.DataFrame) -> dict:
    '''
    Function that creates a dictionary of cluster information to apply association rules to
    Inputs:
    - data_w_clusters (pd.Dataframe): Dataframe indexed by customer_id that includes the associated cluster
        We assume the cluster column is 'cluster_kmeans'
    - encoded_purchases (pd.Dataframe): Dataframe indexed by invoice_id that describes the customers'
    purchases, encoded with transaction encoding

    Outputs:
    - cluster_dfs (dict): Dict of cluster_label | Purchases filtered for that cluster
    '''
    # Joining the cluster labels onto the customer baskets and anonymizing
    basket_rules_df = pd.merge(encoded_purchases,
                            data_w_clusters[['cluster_kmeans']],
                            how='inner', left_on='customer_id', right_index=True)
    basket_rules_df.drop('customer_id', axis=1, inplace=True)

    # Separating the basket_rules_df dataframe into one per cluster
    grouped = basket_rules_df.groupby('cluster_kmeans')
    cluster_dfs = {cluster: data.drop(columns='cluster_kmeans') for cluster, data in grouped}

    return cluster_dfs

def association_rules_apriori(purchases_dict:dict, index:int) -> pd.DataFrame:
    '''
    Function that applies Apriori association rules assuming min_support=0.05
    Inputs:
    - purchases_dict (dict): Dictionary of purchases per cluster
        Created by association_rules_preproc()
    - index (int): Key to look for in the dict
    Outputs:
    - Prints the frequent items and rules created of that cluster
    - Returns the dataframe of rules
    '''
    frequent_itemsets_grocery = apriori(purchases_dict[index], min_support=0.05, use_colnames=True)

    print(frequent_itemsets_grocery.sort_values(by='support', ascending=False))

    rules_grocery = association_rules(frequent_itemsets_grocery,
                                    metric="confidence",
                                    min_threshold=0.2)

    print(f'Cluster {index}:')
    return rules_grocery.sort_values(by='lift', ascending=False)


def association_rules_eclat(purchases_dict:dict, index:int) -> pd.DataFrame:
    '''
    Function that applies ECLAT association rules assuming min_support=0.05
    Inputs:
    - purchases_dict (dict): Dictionary of purchases per cluster
        Created by association_rules_preproc()
    - index (int): Key to look for in the dict
    Outputs:
    - Prints the frequent items and rules created of that cluster
    - Returns the dataframe of rules
    '''
    eclat_groceries = ECLAT(data=purchases_dict[index])
    
    groceries_rules_indexes, groceries_rules_supports = eclat_groceries.fit(min_support=0.05,
                                           min_combination=2,
                                           max_combination=2)
    
    rules_eclat_groceries = pd.DataFrame(list(groceries_rules_supports.values()),
                                    index=list(groceries_rules_supports.keys()),
                                    columns=['support'])

    print(f'Cluster {index}:')
    return rules_eclat_groceries.sort_values(by='support', ascending=False)