# Generate handwritten numbers, letters and shapes by using JointVAE
In the context of the lecture "Deep Learning - Architectures and Methods" (TU Darmstadt), we re-implemented [JointVAE](https://github.com/Schlumberger/joint-vae) to generate handwritten numbers, letters and shapes. 

# Experiments
We tested different dimensionalities for discrete and continuous variables for the different datasets listed in the references. In the following, we only show some interesting results instead of all generated images. 

## Different dimensionalities for latent discrete variables
### Number of discrete values = 1
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_1/all_traversals.png" width="400">
You can see that the latent continuous variables change the classes, which is not desired. In addition no desired aspects like changing rotation or width and others are learned. 


### Number of discrete values = Number of classes / 2
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/shape/discrete/dist2_traversal.gif" width="400">
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_7/all_traversals.png" width="400">
Again not all classes are learned here and the latent continuous variables do change the class rather than performing transformations.


### Number of discrete values = Number of classes
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/shape/discrete/dist4_traversal.gif" width="400">
Mainly two classes, star and circle are learned. Again the latent continuous variables do the job of changing classes and changing things like rotation does often also resolve in changing the class.


<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_14/all_traversals.png" width="400" margin-top:20px>
Again not all classes are learned here, but the continuous latent variables do also learn some desired behavior like changing thickness.


### Number of discrete values = Number of classes x 2
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/shape/discrete/dist8_traversal.gif" width="400">
All classes are represented by atleast one discrete variable and the latent continuous variables do not influence the classes anymore!


<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_28/all_traversals.png" width="400"; padding-top:10px>
All classes are represented by atleast one discrete variable and the latent continuous variables do represent some desired behavior, however some of them do change the classes like from - to a 1, which is however acceptable if you think about rotation!


## Different dimensionalities for latent continuous variables
## Number of continuous values = 1
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_14_cont_1/all_traversals.png" width="400">
Not all classed are learned and the single continuous variable does the job of changing classes and applying transformations. The pattern of changing classes is also seen for the other discrete classes, not shown in the picture above.


### Number of continuous values = 5
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/shape/continous/cont5_traversal.gif" width="400">
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/arabic/continous/cont5_traversal.gif" width="400">
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_14_cont_5/all_traversals.png" width="400">
The continous variables do change classes and transformations at once, however they do learn transformations now instead as it is the case for 1 variable, which mostly just learned changing classes.


### Number of continuous values = 10
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/shape/continous/cont10_traversal.gif" width="400">
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/arabic/continous/cont10_traversal.gif" width="400">
Latent continuous variables start to do their job, however there are some which also change the classes when applying transformations.


### Number of continuous values = 20
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/shape/continous/cont20_traversal.gif" width="400">
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/RESULTS/arabic/continous/cont20_traversal.gif" width="400">
<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/Results_Jascha/ep50_disc_14_cont_20/all_traversals.png" width="400">
It may be the case, that just some of the continuous variables do have a real influence, however you can select the interesting once since they are pretty nicely disentangled.


## Different loss functions 
### MSE
### Binary Cross Entropy
### Cauchy loss
In consultation with our lecturer, we used the [Cauchy Loss](https://ieeexplore.ieee.org/abstract/document/8809729), trying to improve the results. We also used different scale values (in the figure donated as "c") for an approach to improve the results.

<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/RESULTS/CauchyScalarValue.png" width="400">

However, the results became significantly worse for MNIST:

<img src="https://github.com/JuAbels/joint-vae/blob/Tillmann/trained_models/mnist_cauchy_100epochs_scale%3D1_disc%3D10/pictures_from_notebook/generated_samples_fromthemodel.png" width="400">

# References
Barron Jonathan T., “A General and Adaptive Robust Loss Function,” CVPR (2019).

Shi Wei, Xiong Kui, and Wang Shiyuan, Multikernel Adaptive Filters Under the Minimum Cauchy Kernel Loss Criterion, IEEE Access, 2019.

## Code Reference
This repo contains [a forked Pytorch implementation](https://github.com/Schlumberger/joint-vae) of [Learning Disentangled Joint Continuous and Discrete Representations](https://arxiv.org/abs/1804.00104) (NIPS 2018) adjusted by our experimenting code.

## Dataset References
Clarence, Zhao. Handwritten Math Symbol and Digit Dataset, 2020. Accessed July 23, 2020. https://www.kaggle.com/clarencezhao/handwritten-math-symbol-dataset/metadata.

LeCun, Yann, and Cortes Corinna. MNIST Handwritten Digit Database, 2010. Accessed July 23, 2020. http://yann.lecun.com/exdb/mnist/.

Mohamed, Loey. Arabic Handwritten Characters Dataset, 2019. Accessed July 23, 2020. https://kaggle.com/mloey1/ahcd1.

Sachin, Patel. A-Z Handwritten Alphabets, 2018. Accessed July 23, 2020. https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format.

smeschke. Four Shapes, 2017. Accessed July 23, 2020. https://www.kaggle.com/smeschke/four-shapes.

## Training Reference
We used the GPU from [Google Colab](https://colab.research.google.com) for training JointVAE. 
