import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sb

groceryrawstats = pd.read_csv('data\hammer-4-raw.csv')
groceryprodstats = pd.read_csv('data\hammer-4-product.csv')


merged_stats = pd.merge(groceryrawstats,groceryprodstats,left_on='product_id',right_on='id')
merged_stats = merged_stats.drop(['detail_url','sku','upc'],axis=1)
salevals = merged_stats[~merged_stats['old_price'].isna()]
walmartsales = salevals[(salevals['vendor'] == 'Walnart')]

vendor_sales_count = salevals['vendor'].value_counts()

total_products = merged_stats.groupby('vendor').size().reset_index(name='total_products')
sale_products = salevals.groupby('vendor').size().reset_index(name='sale_products')
vendor_sales = pd.merge(total_products, sale_products, on='vendor', how='left')
vendor_sales['sales_ratio'] = vendor_sales['sale_products'] / vendor_sales['total_products']

plt.figure(figsize=(10, 6))
sb.barplot(x=vendor_sales_count.index, y=vendor_sales_count.values, palette="viridis")
plt.title("Number of Sales by Vendor")
plt.xlabel("Vendor")
plt.ylabel("Number of Sales")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
sb.barplot(x='sales_ratio', y='vendor', data=vendor_sales.sort_values('sales_ratio', ascending=False), palette='viridis')
plt.title('Ratio of Sales to Total Product Listings by Vendor')
plt.xlabel('Sales to Total Products Ratio')
plt.ylabel('Vendor')
plt.tight_layout()
plt.show()