# Coding Kendall's Shape Trajectories for 3D Action Recognition 

The official PyTorch implementation of "[Coding Kendall's Shape Trajectories for 3D Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tanfous_Coding_Kendalls_Shape_CVPR_2018_paper.html) " (CVPR 2018). 

In addition to the Sparse Coding and classification pipeline described in the paper, this code includes a data augmentation framework based on GANs to generate new sparse codes. This step further improved experimental results on the three datasets. 

## Summary

> Suitable shape representations as well as their temporal evolution, termed trajectories, often lie to non-linear manifolds. This puts an additional constraint (ie, non-linearity) in using conventional machine learning techniques for the purpose of classification, event detection, prediction, etc. This paper accommodates the well-known Sparse Coding and Dictionary Learning to the Kendall's shape space and illustrates effective coding of 3D skeletal sequences for action recognition. Grounding on the Riemannian geometry of the shape space, an intrinsic sparse coding and dictionary learning formulation is proposed for static skeletal shapes to overcome the inherent non-linearity of the manifold. As a main result, initial trajectories give rise to sparse code functions with suitable computational properties, including sparsity and vector space representation. To achieve action recognition, two different classification schemes were adopted. A bi-directional LSTM is directly performed on sparse code functions, while a linear SVM is applied after representing sparse code functions using Fourier temporal pyramid. Experiments conducted on three publicly available datasets show the superiority of the proposed approach compared to existing Riemannian representations and its competitiveness with respect to other recently-proposed approaches. When the benefits of invariance are maintained from the Kendall's shape representation, our approach not only overcomes the problem of non-linearity but also yields to discriminative sparse code functions.

### Requirements

```
pytorch(0.3.1)
torchvision(0.2.0)
keras
cvxpy
numpy
scipy
joblib
```

### Citation 
Please consider citing this work if you use our code in your research:


``` 
@inproceedings{ben2018coding,

  title={Coding Kendall's shape trajectories for 3D action recognition},
  
  author={Ben Tanfous, Amor and Drira, Hassen and Ben Amor, Boulbaba},
  
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  
  pages={2840--2849},
  
  year={2018}
  
}
```