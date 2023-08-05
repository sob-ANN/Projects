# Deep-Learning-APL745-
Machine Learning/Deep Learning Projects done during the course "Deep Learning for Mechanics"-APL745, taught in the Spring '23 semester at IIT, Delhi.

## [Non-Linear Regression](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Non-Linear%20Regression.ipynb)
Polynomial Regression applied to solve a simple problem of projectile motion. Use of Batch/Mini-batch/Stocastic Gradient Descent from scratch and plotting Number of Epochs vs Cost Function plots. Also, using a variable learning rate optimized by Line Search Algorithm

## [Multivariate Regression](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Multivariate%20Linear%20Regression.ipynb)
Multivariate Regression using Gradient Descent to predict house prices.

## [Binary Classification](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Binary%20Classification-Logistic%20Reg.ipynb)
Binary Classification of data containing multiple features which affect rainfall. Using that data we make predictions on a test set whether it will rain tomorrow or not.
We use logistic Regression as our hypothesis function and Binary Cross Entropy loss. Optimized using scipy library.

## [Multi-Class Classification](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/OnevRest%20Classification.ipynb)
Multiclass Classification of MNIST Dataset from scratch. Optimization is done using Scipy Library. Binary Cross Entropy is the loss function used with Logistic Sigmoid as our hypothesis function. Target labels are one-hot encoded to be able to perform matrix operations.

## [Softmax Multi-Class Classification](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Softmax%20Classification.ipynb)
Multi-class Classification using softmax on the MNIST Fashion Dataset from scratch. Optimized using Batch Gradient Descent. Also, the accuracy is compared with the One-vs-Rest case.

## [Neural Network from Scratch](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Neural%20Network%20from%20Scratch.ipynb)
A multi-layer Perceptron or Neural Network was implemented using Numpy only. This includes implementing the forward methods and the gradients of each of the layers, including the activations.
This gives a clear understanding of how Pytorch/Tensorflow work in the background.

## [Convolution Neural Network from Scratch](https://github.com/sob-ANN/Deep-Learning-APL745-/tree/main/CNN%20Working)
Convolution Neural Network(CNN) has been written from scratch. This project includes writing the convolution forward pass, maxpool, implementing the softmax activation and writing their gradients so that backprop can be performed. 

## [Forward Problem using Physics-Informed-Neural-Network](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/forward_problem_main.ipynb)
Physics Informed Neural Networks are a type of Universal Function Approximators that can be used to solve the underlying Differential Equation of a Physics problem. In this example, the deflection of a 1D Bar is modelled using PINN. Deflection at the two boundaries is zero and has been included in the modified loss function. In this case, no Input(x)-Output(displacement(x)) data is given. We have used only the Differential Equation of the underlying Physics and sampled points within the domain and minimised the 'residue' from the Differential Equation.

## [Inverse Problem using Physics-Informed-Neural-Network](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/PINN_bar_inverse_main.ipynb)
In this example, we have solved an inverse problem of the same bar taken above. In this case, the deflections are known at each point on the bar. However, we do not know the physical properties of the bar(Axial Stiffness). Following similar principles as above, we have solved for EA(Axial Stiffness).

## [2D PINN](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/2D%20PINN%20Project.ipynb)
Solution of a 2-Dimentional Elastic Deformation problem using Physics-Informed-Neural-Network. The solution is obtained by minimising the 'residue' of the governing Partial Differential Equations while also respecting the Direchlet Boundary Conditions. Further, a number of test body forces are used in order to see the different responses.
## [Deep-O-Net](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Deep_o_net_Final.ipynb)
Here we dive into Operator Learning. Deep-O-Net has been implemented using Pytorch. Data is generated in the same way as was done in the paper(https://arxiv.org/abs/1910.03193) for the Antiderivative operator.
## [Fourier Neural Operator](https://github.com/sob-ANN/Deep-Learning-APL745-/blob/main/Fourier%20Neural%20operator.ipynb)
Applying a Fourier Neural Operator on 1d Burgers' Equation as discussed in the paper (https://scholar.google.com/scholar_lookup?arxiv_id=2010.08895)
