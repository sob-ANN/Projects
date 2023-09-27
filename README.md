# My Portfolio

Welcome to my GitHub repository, where I showcase my projects in various domains. Below, you'll find a categorized list of my work along with links to specific projects and descriptions.

## Table of Contents

- [Bayesian Inference](#bayesian-inference)
- [Data Science Related](#data-science-related)
- [Machine Learning From Scratch](#machine-learning-from-scratch)
- [Operator Learning](#operator-learning)
- [Physics-Informed Neural Networks](#physics-informed-neural-networks)
- [Math-Related Projects](#math-related-projects)

---

## Bayesian Inference

Explore my projects related to Bayesian Inference, where I leverage probabilistic modeling and statistical methods to solve various problems.

### Bayesian Coin Tossing

- [Project Link](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Coin%20Tossing%20(Probabilistic).ipynb)
- Description: A simple coin tossing example modeled using Bayesian Statistics with Pymc3. Explore the No U-Turn Sampling (NUTS) technique for posterior approximation and visualize the Binomial distribution with different parameters.

### Approximate Integration

- [Project Link](./Bayesian%20Inference/Approximate%20Integration.ipynb)
- Description: Discover how I perform integration using Monte-Carlo Estimation and compare results with true values. Additionally, compare Gaussian Quadrature Methods for calculating Entropy of a PDF with Monte Carlo Estimates.

### Integration involving PDFs

- [Project Link](./Bayesian%20Inference/Integration%20involving%20PDFs.ipynb)
- Description: Learn about integration (expectation) of the product of Probability Distribution Functions using Gaussian Quadrature, and how it can be used to recover mean and variance.

### Bayesian Neural Network in TensorFlow

- [Project Link](./Bayesian%20Inference/Bayesian%20Neural%20Network%20in%20TensorFlow.ipynb)
- Description: Dive into the implementation of Bayesian Neural Networks using TensorFlow API, modeling both Epistemic and Alleatoric Uncertainty. Explore inference with Bayes by Backprop using TensorFlow Probability Layers.

### Parameter Estimation using MCMC

- [Project Link](./Bayesian%20Inference/Parameter%20Estimation%20using%20MCMC.ipynb)
- Description: Explore parameter estimation for a single-degree-of-freedom Structural Dynamical System using Markov Chain Monte Carlo (MCMC) and State-Space modeling.

### Kalman Filter State Estimation

- [Project Link](./Bayesian%20Inference/Kalman%20Filter%20State%20Estimation.ipynb)
- Description: Learn about state estimation of a single-degree-of-freedom Structural Dynamical System using Kalman Filter and state-space formulation.

### Probabilistic Kalman Filter

- [Project Link](./Bayesian%20Inference/Probabilistic%20Kalman%20Filter.ipynb)
- Description: Discover the probabilistic form of Kalman Filter and its applications in simplifying complex integrations for non-Gaussian assumptions.

For questions or collaboration opportunities, you can reach out to me at [your.email@example.com](mailto:your.email@example.com).

---

## Data Science Related

### Data-Driven Fantasy Premier League (FPL) Team Selection

In the world of Fantasy Premier League (FPL), where over 10 million football fans compete annually, data-driven decisions can make all the difference. In this project, I harnessed data from the FPL website using an API call to gain a competitive edge in team selection. Here's what I explored:

- **Metrics for Player Selection**: I created custom metrics such as 'Value' (points scored/cost) to identify cost-effective players. Additionally, I delved into Expected Goals (xG) and Expected Assists (xA) to uncover players who outperformed their xG, offering insights into potential overperformers and underperformers.

- **Beyond the Numbers**: While data is essential, sometimes players can defy statistical expectations. I delved into the fascinating world of "looking beyond the underlying numbers" to evaluate player performance beyond mere statistics. 

- **Optimal Team Building**: I attempted to find the 'most optimal' FPL team while navigating constraints such as budget limitations, maximum/minimum players per position, and team quotas. This project was my first step in using data to optimize team selection.

I plan to extend this project in the future to include in-season changes and predictions. Historical data may also play a crucial role in making informed decisions. This was my initial attempt to apply my learning and passion for FPL in a data-driven manner.

### Sentiment Analysis with RNN and LSTM

[Project Description Goes Here]

### Neural Network with TensorFlow

Explore the fundamentals of Neural Networks with this basic implementation using TensorFlow. I applied this model to the MNIST dataset to understand the representations learned by hidden layers while mastering TensorFlow's API.

### Support Vector Machine

Dive into the world of classification with the Support Vector Machine (SVM) algorithm, implemented using the sklearn library. This project explores various kernel functions, including radial basis function (rbf) and polynomial kernels, with extensive hyperparameter tuning to achieve high accuracy. You can also check out a Kaggle competition where my model achieved an impressive accuracy of 0.98 ([Kaggle Competition Link](https://www.kaggle.com/competitions/ell-784-assignment-2/leaderboard)).

### K-means Clustering

Uncover patterns in data with the K-means Clustering algorithm. This unsupervised clustering model was applied to the MNIST dataset using sklearn. I experimented with different values of 'k' (number of clusters) to understand how it impacts prediction accuracy.

### KLD and VAE

[Project Description Goes Here]

---

## Machine Learning From Scratch

### Non-Linear Regression

- **Polynomial Regression**: Projectile motion problem solved with Batch, Mini-batch, and Stochastic Gradient Descent, plus cost function plots and variable learning rates.

### Multivariate Regression

- Predict house prices using Gradient Descent with multiple features.

### Binary Classification

- Rainfall prediction with logistic Regression, Binary Cross-Entropy loss, and scipy optimization.

### Multi-Class Classification

- MNIST Dataset classification using Binary Cross-Entropy loss and one-hot encoding.

### Softmax Multi-Class Classification

- MNIST Fashion Dataset classification with Batch Gradient Descent and accuracy comparison.

### Neural Network from Scratch

- Numpy-based multi-layer perceptron for deep learning insights.

### Convolution Neural Network from Scratch

- CNN built from scratch with convolution, max-pooling, softmax, and gradients.

### Recurrent Neural Network from Scratch

- RNN mathematically implemented in Numpy with Tanh activation.

---

## Operator Learning

### Deep-O-Net

Dive deep into Operator Learning with Deep-O-Net, implemented using PyTorch. Generate data using methods similar to those outlined in the paper ([Paper Link](https://arxiv.org/abs/1910.03193)) for the Antiderivative operator.

### Fourier Neural Operator

Explore the application of a Fourier Neural Operator on 1D Burgers' Equation, as discussed in the paper ([Paper Link](https://scholar.google.com/scholar_lookup?arxiv_id=2010.08895)).

---

## Physics-Informed Neural Networks

### Forward Problem using Physics-Informed-Neural-Network

Discover the power of Physics-Informed Neural Networks (PINN) in solving forward problems. In this project, we model the deflection of a 1D bar using PINN. Unlike traditional methods, we don't rely on input-output data but instead leverage the underlying physics represented by differential equations. By sampling points within the domain and minimizing the 'residue' from the differential equation, we uncover valuable insights into the deflection of the bar. Boundary conditions, where deflection is zero, are also incorporated into the modified loss function.

### Inverse Problem using Physics-Informed-Neural-Network

Take on the inverse problem of the same bar modeled in the previous project. In this case, we have deflection data for each point on the bar, but we lack information about the physical properties of the bar, specifically the Axial Stiffness (EA). Applying similar principles as in the forward problem, we solve for EA (Axial Stiffness) using PINN.

### 2D PINN

Explore the solution of a 2-Dimensional Elastic Deformation problem using Physics-Informed Neural Networks. We achieve this by minimizing the 'residue' of the governing Partial Differential Equations while adhering to Dirichlet Boundary Conditions. To analyze different responses, a variety of test body forces are applied to the system, providing comprehensive insights into the 2D Elastic Deformation problem.

---

## Math-Related Projects

### Advection Diffusion

Explore the solution of the Advection Diffusion Partial Differential Equation (PDE) on real-world wind data. This project investigates different scenarios by varying the "Peclet Number," providing insights into how advection and diffusion processes interact.

### Image Convolution using Matrices

Dive into the world of image processing and convolution operations with matrices. Discover how various filters are applied to images through matrix operations, offering a unique perspective on image manipulation techniques.

### Partial Differential Equation

Solve a Partial Differential Equation (PDE) with Dirichlet Boundary Conditions and visualize the results. This project offers a visual representation of the solution space, making it easier to comprehend and interpret the outcomes.

### Projectile Motion Using odeint

Explore the dynamics of projectile motion in this project, leveraging Scipy's 'odeint' library to solve differential equations. Witness the motion of a football launched from the ground while accounting for drag forces. Realistic parameter values are employed to simulate the motion accurately.

---

## Contact

If you have any questions or would like to get in touch, feel free to reach out to me at [sobanlone88@gmail.com](mailto:sobanlone88@gmail.com).

