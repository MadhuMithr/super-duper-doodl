import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df=pd.read_csv("C:\\Users\\MADHU MITHRA\\Downloads\\archive (1)\\Mall_Customers.csv")
k=6
print(df)
x=df[['Annual Income (k$)','Spending Score (1-100)']]
kmeans=KMeans(k)
kmeans.fit(x)
labels=kmeans.labels_
col = {0: 'r', 1: 'g', 2: 'b', 3: 'c', 4: 'm', 5: 'y'}
for i in labels:
    plt.scatter(x.iloc[labels==i,0],x.iloc[labels==i,1],color=col[i])
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
