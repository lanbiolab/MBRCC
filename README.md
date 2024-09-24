# Contrastive Clustering Learning for Multi-Behavior Recommendation
 Increasing multiple behavior recommendation models have achieved great successes. However, many models do
 not consider commonalities and differences between behaviors and data sparsity of the target behavior. This
 paper proposes a novel Multi-behavior Recommendation Model (MBRCC) based on Contrastive Clustering
 Learning. Specifically, the graph convolutional network(GCN) is employed to obtain the embeddings of
 users and items, respectively. Then, three kinds of tasks (including behavior-level embedding, instance-level
 embedding and cluster-level embedding) are designed to optimize the embeddings of users and items. In
 behavior-level embedding, we design an adaptive parameter learning strategy to analyze the impact of auxiliary
 behaviors on the target behavior. Then, the embeddings of users for each behavior are weighted to obtain the
 final embeddings of users. In instance-level embedding, we employ contrastive learning to analyze the instances
 of user and item for mitigating the issue of data sparsity. In cluster-level embedding, we design a new cluster
 contrastive learning method to capture the similarity between groups of user and item. Finally, we combine these
 three tasks to improve the quality of the embeddings of users and items. We conduct extensive experiments on
 three real-world datasets and experimental results indicate that the MBRCC remarkably outperforms numerous
 existing recommendation models.
