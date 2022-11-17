import pandas as pd
import csv
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori,association_rules

dataset = []
with open('Market_Basket_Optimisation.csv') as file:
  reader = csv.reader(file, delimiter=',')
  for row in reader:
    dataset += [row]

dataset[1:10]

len(dataset)

te= TransactionEncoder()
x=te.fit_transform(dataset)

df = pd.DataFrame(x,columns=te.columns_)

len(te.columns_)

df.head()

#find frequent itemsets
freq_itemset = apriori(df,min_support=0.01,use_colnames=True) 

#find the rules
rules=association_rules(freq_itemset,metric='confidence',min_threshold=0.25)

rules=rules[['antecedents','consequents','support','confidence']]

rules[rules['antecedents']=={'cake'}]['consequents']
