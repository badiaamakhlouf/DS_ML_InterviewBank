# ML : Unsupervised learning 

Machine learning algorithms can be categorized into different types based on their learning approaches, data characteristics, and task objectives. Unsupervised learning is one such type where the models are trained on unlabeled data to uncover hidden patterns or structures without explicit guidance. It's about finding relationships or structures within data without predefined outcomes or target variables.

💡 Unsupervised learning encompasses two main techniques:
   - Clustering
   - Associations

🧠 This GitHub page provides essential information and detailed insights into each technique, offering comprehensive and tailored questions for technical interviews for data scientists and machine learning engineers. It can help you master everything about Clsutering and associations, ensuring you're well-prepared to pass your technical interview.

🙏🏻 Note: Your suggestions for improvement are welcome; please feel free to provide feedback to help maintain the precision and coherence of this page.

## List Of Questions :
### Q0- What are Unsupervised Learning techniques ?
- We have two techniques, Clustering and association:
    - **Custering :** involves grouping similar data points together based on inherent patterns or similarities. Example: grouping customers with similar purchasing behavior for targeted marketing.
    - **Association :** identifying patterns of associations between different variables or items. Example: e-commerse website suggest other items for you to buy based on prior purchases.

### Q1- What does Clustering Analysis mean?
- It is unsupervised machine learning technique to group similar data points based on their characteristics.
- It aims to identify clusters within a dataset.
- Data points within the same cluster are more similar to each other than to those in other clusters.
- Clustering analysis is commonly used for:
   - Exploratory data analysis
   - Pattern recognition
   - Segmentation 
- It is used in various fields such as:
   - Marketing
   - Biology
   - Finance
   - etc.
- Sometimes, you know the number of clusters and sometimes you must determine it.
- More details and additional questions regarding clustering analysis are available in the subsequent page : [Clustering Analysis](./clustering_analysis.md)

### Q1- What does Association means?
- Known as association rule mining
- It is a rule-based approach used to discover interesting relationships between features in a given dataset.
- It works by using a measure of interest to identify strong rules found within a dataset. 
- Application: 
   - **Retail:** Understanding purchasing patterns for better product placement.
   - **Recommendation Systems:** Suggesting related items based on past user behavior.
   - **Healthcare:** Finding correlations between symptoms and diseases for diagnosis and treatment planning.
   - **Web Usage Mining:** Analyzing user behavior to improve website experience.
- Most common algorithms:
   - Apriori Algorithm: A classic algorithm that discovers frequent itemsets and generates association rules.
   - FP-Growth (Frequent Pattern Growth): An efficient algorithm for discovering frequent itemsets using a divide-and-conquer strategy.
   - Eclat (Equivalence Class Transformation): Another frequent itemset mining algorithm that uses a depth-first search strategy to find frequent itemsets.
   - FPMax (Maximal Frequent Pattern): An extension of FP-Growth that extracts maximal frequent itemsets.
   - CARMA (Class Association Rule Mining Algorithm): An algorithm designed specifically for mining class association rules, which consider both itemset frequency and class labels.





