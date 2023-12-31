## [Bayesian Coin Tossing](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Coin%20Tossing%20(Probabilistic).ipynb)
A simple coin tossing example modelled using Bayesian Statistics implemented in Pymc3. Posterior is approximated using the No U-Turn Sampling (NUTS) and the traces are plotted using ArviZ. Further, multiple plots help visualize the Binomial distribution with different parameters.

## [Variational Auto Encoder](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/Variational%20AutoEncoder%20Tensorflow.ipynb)
Jumping into Generative AI, here I implement Variational AutoEncoder of Fashion MNIST Data. Building the VAE by minimizing the ELBO (Evidence Lower BOund), I encode the information using deep NNs and later decode it using the same. This simple implementation is used to generate new data points by constructing this Encoder-Decoder Sequence.


## [Approximate Integration](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Numerical%20Integration%20(MC,%20Gauss%20Quadrature).ipynb)
I have taken a few examples where integration has been performed using Monte-Carlo Estimation and compared with the true values. Further, I have compared Gaussian Quadrature Methods for calculating Entropy of a PDF with the Monte Carlo Estimates. Other examples caluclate different Expectation values for which we already know a closed form solution which is compared with the two methods.

## [Integration involving PDFs](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Integration%20of%20PDFs%20Gauss%20Quadrature.ipynb)
Integration(Expectation) of product of Probability Distribution Functions with some functions using Gaussian Quadrature. Functions are taken for which a closed form solution already exists. We can recover the mean(first moment), variance (second moment) using this method.

## [Bayesian Neural Network in TensorFlow](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Bayesian%20Neural%20Network%20Tensorflow.ipynb)
Bayesian Neural Network(BNN) implementation using TensorFlow API. Here I have taken linear and non-linear regression data with noise and have fit a BNN with both a Deterministic output (modelling only Epistemic Uncertainty) and a Normal Distribution as output (modelling both Epistemic and Alleatoric Uncertainty). Inference is performed using Bayes by Backprop method which is built-in in the DenseVariational Layer in the TensorFlow Probability Layers API.

## [Parameter Estimation using MCMC](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Parameter%20Est%20MCMC%20SDOF.ipynb)
Parameter Estimation of a single-degree-of-freedom Structural Dynamical System using Markov Chain Monte Carlo (MCMC). System is modelled as a State-Space. We calculate the energy (-loss) of the model and update the parameters using sampling based Metropolis Hastings Algorithm. 

## [Kalman Filter State Estimation](https://github.com/sob-ANN/Projects/blob/main/Bayesian/State%20Est%20SDOF%20KF.ipynb)
State Estimation of a single-degree-of-freedom Structural Dynamical System using Kalman Filter. This involves generating data, predicting the states (displacement and velocity) using measued accelaration data(noisy) only. Again, a state-space formulation has been used.



## [Probabilistic Kalman Filter](https://github.com/sob-ANN/Projects/blob/main/Bayesian/Probabilistic%20Kalman%20Filter.ipynb)
Kalman Filter, one of the most widely used filtering algorithm, in it's probabilistic form, has some expressions in the derivation. The main assumption of Kalman Filter is that the predicted states are Gaussian. Using this idea, the complex integrations are simplified. Here, I try to recreate some of the results using approximations which will help us in expanding to non-Gaussian assumptions as well. Most of the equations have been taken from Chapter 4 of the book [Bayesian Filtering and Smoothing, Simo Särkkä
](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwiEkrDm_MCBAxXY3jgGHYbHDBYQFnoECCEQAQ&url=https%3A%2F%2Fusers.aalto.fi%2F~ssarkka%2Fpub%2Fcup_book_online_20131111.pdf&usg=AOvVaw2N7Ex3iUkENBRwcn8_0_LU&opi=89978449) . I have given the mathematical expressions used in the Notebook Markdown.


## [Kullback-Leiber Divergence Minimisation Visualised](https://github.com/sob-ANN/Projects/blob/main/Data%20Science%20Related/KL_divergence%20Tensorflow.ipynb)
I look at minimisation of KL Divergence between two distributions and the effect of using Reverse KLD (As used in Expectation Propagation). There's a nice visualisation which shows the 2D Gaussian moving towards the intended distribution in both examples.

