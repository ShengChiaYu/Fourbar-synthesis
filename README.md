# A Generative Model for Path Synthesis of Four-bar Linkages via Uniformly Sampled Dataset

## Abstract
Solving the path synthesis of four-bar linkages via optimization algorithm may encounter issues as defects and guess of initial values. Recently, the method for resolving such problems associated with deep-learning schemes shows that it can avoid these issues. However, there is still room for improving the accuracy of the scheme. In this work, a framework including preprocessing, data generation, and model training for the path synthesis of four-bar linkage are presented. The preprocess starts by regenerating the target path with evenly distributed points along the path, followed by the normalization of the shape and feature extraction. For data generation, an unsupervised learning is employed to adjust the distribution of paths of different shapes in the dataset so that the robustness of the model can be achieved. As for model training, models based on datasets of different types of four-bar linkages as well as a classifier to determine the suitable generative model for the target path is constructed. Finally, several examples including closed and open paths are illustrated to verify the effectiveness of the framework.  

**Keywords**: planar four-bar linkage, path synthesis, machine learning, generative model, data generation, mechanism synthesis

