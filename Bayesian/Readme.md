## [Bayesian Coin Tossing](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Coin%20Tossing%20(Probabilistic).ipynb)
A simple coin tossing example modelled using Bayesian Statistics implemented in Pymc3. Posterior is approximated using the No U-Turn Sampling (NUTS) and the traces are plotted using ArviZ. Further, multiple plots help visualize the Binomial distribution with different parameters.

## [Approximate Integration](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Numerical%20Integration%20(MC,%20Gauss%20Quadrature).ipynb)
I have taken a few examples where integration has been performed using Monte-Carlo Estimation and compared with the true values. Further, I have compared Gaussian Quadrature Methods for calculating Entropy of a PDF with the Monte Carlo Estimates. Other examples caluclate different Expectation values for which we already know a closed form solution which is compared with the two methods.

## [Integration involving PDFs](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Integration%20of%20PDFs%20Gauss%20Quadrature.ipynb)
Integration(Expectation) of product of Probability Distribution Functions with some functions using Gaussian Quadrature. Functions are taken for which a closed form solution already exists. We can recover the mean(first moment), variance (second moment) using this method.

## [Bayesian Neural Network in TensirFlow](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Bayesian%20Neural%20Netwoek%20Tensorflow.ipynb)
Bayesian Neural Network(BNN) implementation using TensorFlow API. Here I have taken linear and non-linear regression data with noise and have fit a BNN with both a Deterministic output (modelling only Epistemic Uncertainty) and a Normal Distribution as output (modelling both Epistemic and Alleatoric Uncertainty). Inference is performed using Bayes by Backprop method which is built-in in the DenseReparameterization Layer in the TensorFlow Probability Layers API.
