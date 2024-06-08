from pyECLAT import ECLAT
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pandas as pd

def association_rules_preproc(info_treated, basket_encoded):
    # Joining the cluster labels onto the customer baskets and anonymizing
    basket_rules_df = pd.merge(basket_encoded,
                            info_treated[['cluster_kmeans']],
                            how='inner', left_on='customer_id', right_index=True)
    basket_rules_df.drop('customer_id', axis=1, inplace=True)

    # Separating the basket_rules_df dataframe into one per cluster
    grouped = basket_rules_df.groupby('cluster_kmeans')

    # cluster_dfs - dictionary of the form cluster label : Dataframe filtered for that cluster
    cluster_dfs = {cluster: data.drop(columns='cluster_kmeans') for cluster, data in grouped}

    return cluster_dfs

def association_rules_apriori(cluster_dfs, index):
    frequent_itemsets_grocery = apriori(cluster_dfs[index], min_support=0.05, use_colnames=True)

    print(frequent_itemsets_grocery.sort_values(by='support', ascending=False))

    rules_grocery = association_rules(frequent_itemsets_grocery,
                                    metric="confidence",
                                    min_threshold=0.2)

    return print(f'Cluster {index}:' + rules_grocery.sort_values(by='lift', ascending=False))

def association_rules_eclat(cluster_dfs, index):
    eclat_groceries = ECLAT(data=cluster_dfs[index])
    
    groceries_rules_indexes, groceries_rules_supports = eclat_groceries.fit(min_support=0.02,
                                           min_combination=2,
                                           max_combination=2)
    
    rules_eclat_groceries = pd.DataFrame(list(groceries_rules_supports.values()),
                                    index=list(groceries_rules_supports.keys()),
                                    columns=['support'])

    return rules_eclat_groceries.sort_values(by='support', ascending=False).head(10)