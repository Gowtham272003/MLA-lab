import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Create sample sales data
data = {
    'CustomerID': [1, 1, 1, 2, 2, 3, 3, 4, 5, 5],
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Laptop', 'Keyboard', 'Mouse', 'Keyboard',
                'Laptop', 'Mouse', 'Keyboard']
}
df = pd.DataFrame(data)
print("Sample Sales Data:")
print(df)

# Step 2: Add a column to indicate a purchase
df['Bought'] = 1

# Step 3: Create a customer-product matrix (1 if bought, 0 if not)
customer_product_matrix = df.pivot_table(index='CustomerID', columns='Product',
                                        values='Bought', aggfunc='max', fill_value=0)
print("\nCustomer-Product Matrix:")
print(customer_product_matrix)

# Step 4: Calculate similarity between products using cosine similarity
product_similarity = cosine_similarity(customer_product_matrix.T)
similarity_df = pd.DataFrame(product_similarity, index=customer_product_matrix.columns,
                              columns=customer_product_matrix.columns)
print("\nProduct Similarity Matrix:")
print(similarity_df)

# Step 5: Recommend similar products
def recommend_products(product_name, similarity_matrix, top_n=3):
    if product_name not in similarity_matrix.columns:
        print(f"Product '{product_name}' not found.")
        return []
    
    # Sort the product similarity scores in descending order
    similar_scores = similarity_matrix[product_name].sort_values(ascending=False)
    
    # Exclude the product itself from the recommendations
    recommendations = similar_scores.iloc[1:top_n+1].index.tolist()
    return recommendations

# Step 6: Example usage
product_to_check = 'Keyboard'
recommendations = recommend_products(product_to_check, similarity_df)
print(f"\nRecommended products for '{product_to_check}':")
print(recommendations)

# Step 7: Visualize the similarity matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_df, annot=True, cmap="YlGnBu")
plt.title("Product Similarity Heatmap")
plt.show()
