# DEXTER
Age and Gender prediction using Wide Residual Networks

## Abstract
In this project, we introduce a deep learning solution
to age and gender estimation from a single face image. For both
prediction tasks, a single Convolutional Neural Network (CNN) of
Wide Residual Network (WRN-16-8) architecture is used. We
pose the age estimation problem as a deep classification problem
followed by a softmax expected value refinement. The network
was trained from scratch on IMDB faces and tested on WIKI
faces from the IMDB-WIKI dataset.

| MAE (Age) | Accuracy (Gender) | Min. Face Score | # Test images |
|-----------|-------------------|-----------------|---------------|
| 4.46      | 97.60%            | 6.0             | 250           |
| 4.89      | 97.14%            | 5.0             | 3184          |
| 5.38      | 96.80%            | 4.0             | 11312         |
| 6.01      | 96.37%            | 3.0             | 21608         |

Technical report: [Link](https://github.com/rachitrawat/DEXTER/blob/master/report.pdf)
