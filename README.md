# TSVM
For Functional Non-coding Variants Prediction using genomic sequence
# Introduction
We propose TSVM to predict context-specific functional NCVs, Which uses a convolutional neural network to extract features and support vector machine to predict functional non-coding variants. First, the convolutional layer was pretrained using large-scale generic functional non-coding variants as sequence feature extractors.  Second, a combination of random forest and support vector machine was used to predict context-specific functional non-coding variants.  
# Requirements
● Python 3.8                                                                                                                                                               ● Keras == 2.4.0
● TensorFlow == 2.3.0
● numpy >= 1.15.4
● scipy >= 1.2.1
● scikit-learn >= 0.20.3
