# Clustering-Categorical-Data

The problem of clustering becomes more challenging when the data is categorical, that is, when there is no inherent distance measure between data values. This is often the case in many domains, where data is described by a set of descriptive attributes, many of which are neither numerical nor inherently ordered in any way. Moreover, clustring categorical sequences is a challenging problem due to one more reason the difficulties in defining an inherently meaningful measure of similarity between sequences. Cluster validation, which is the process of evaluating the quality of clustering results, plays an important role for practical machine learning systems. Categorical sequences, such as biological sequences in computational biology, have become common in real-world applications. Without a measure of distance between data values, it is unclear how to define a quality measure for categorical clustering. To do this, we employ mutual information, a measure from information theory. A good clustering is one where the clusters are informative about the data objects they contain. Since data objects are expressed in terms of attribute values, we require that the clusters convey information about the attribute values of the objects in the cluster, 
	The evaluation of sequences clustering is currently difficult due to the lack of an internal validation criterion defined with regard to the structural features hidden in sequences. To solve this problem, a novel cluster validity index (CVI) is proposed as a function of clustering, with the intra-cluster structural compactness and inter-cluster structural separation linearly combined to measure the quality of sequence clusters. Cluster validation, which is the process of evaluating  the quality of clustering results, plays an important role in many
issues of cluster analysis A partition-based algorithm for robust clustering of categorical sequences and the CVI are then assembled within the common model selection procedure to determine the number of clusters in categorical sequence sets. Currently, cluster  validation remains an open  problem due to the unsupervised nature of clustering tasks, where no external validation criterion is available to evaluate the result. Despite different aspects of cluster validation , we are interested in the problem of determining the optimal number of clusters in a data set , as most basic clustering algorithms assume that the number of clusters is a user-defined parameter, which, however, is difficult to set in practice.

Algorithmic Description Of Methods 
A . Robust K-means for sequences  
Input: Categorical dataset and initial value of cluster k;
Output: The set of resulting clusters C being the number of nonnoise clusters. 
Begin
       1. Convert the categorical dataset into binary dataset using latine-1 encoding method
       2. Apply the PCA algorithm on binary data set plant_df .
       3. Let q be the number of iterations , q=0. Denote the initial clustering of  S by C(0) = ø;
       4. Let K  be the number of resulting clusters,  K= Ḱ ;
       5. Repeat
              5.1 q=q+1 ;
              5.2 Generate K  clusters by assigning each element in dataset to its nearby cluster.
                     Denote the new clustering by  C(q) ;
              5.3 Remove K- Dependent noise from  C(q) by calling the  Noise Detection Algorithm 
                    and update the number of cluster   K = | C(q) |.
              5.4 Recompute the centroid for each cluster in V  by averaging the vectors belonging 
                    to that cluster.
       until C(q) = C(q-1) ;
       6. Output C(q) ,
 End 
B. Noise Detection Routine
Input: C
Begin 
        1. Let   O = {Ck | Ck ϵ C and  nk = 1}. If O = ø then go to step 3.
        2. Otherwise, choose one cluster from O  and delete it from C  ; assign each object in   
             that cluster to its closest cluster and return;
        3. for each Ck  ϵ C  satisfying  nk  ≤ τ do
                3.1 Create C' by deleting Ck   from C  and assigning the objects in Ck to their 
                      respective closest cluster ;
                3.2  If  CVIc(C') ≤ CVIc(C) then replace C  with C' and return.
        End
End

