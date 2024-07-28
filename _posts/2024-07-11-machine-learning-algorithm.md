---
layout: post
title: Machine Learning Algorithm
date: 2024-07-12 11:12:00-0400
description: ML algorithm
categories: MachineLearning
tags: MachineLearning
---

- statistical perspective: $$Y=f(X)+\epsilon$$ 
  - model=algorithm(data)
- computer science perspective:
  - output = program(input)
  - prediction = program(instance)

- algorithms that learn a mapping from input to output
  - ML algorithms are techniques for estimating target function f to predict the output (Y) given input variables (X)
  - different ML algorithms make different assumptions about the shape and structure of the target function and how best to optimize a representation to approximate it, which makes fitting and comparing a suite of algorithms necessary.

## Algorithms & Models

- linear models
  - linear regression
  - generalized linear models
  - logistic regression (LR)
  - linear discriminant analysis (LDA)
- tree-based methods
  - decision trees
  - random forest
  - gradient boosting
- instance-based methods
  - kNN
  - SVM
- neural networks
  - feedforward neural networks
  - convolutional neural networks
  - recurrent neural networks
  
### Binary classification


- input vector $$x \in \mathbb{R}^d$$
- output $$ y \in \{0,1\} $$
- the goal is to construct a function \begin{equation}f: \mathcal{X} \rightarrow \{0,1\} \end{equation} 

using 0-1 loss, the risk of a classifier $$ f: \mathcal{X} \rightarrow Y $$ is given by:

\begin{equation}
R(f)=EPE(f)E_{X,Y} \mathcal{1}(Y \ne f(X)) = P(Y \ne f(X))
\end{equation}

the Bayes rule $$f^{*}$$ relies on the posterior probabilities

\begin{equation}
f^{*}=\arg \min R(f)=\begin{cases} 1 \quad if \quad P(Y=1|X=x) > P(Y=0|X=x) \\
0 \quad if \quad P(Y=1|X=x) <> P(Y=0|X=x)
\end{cases}
\end{equation}

the Bayes risk is defined as the risk of $$f^{*}$$, which has the smallest possible risk among all possible classifiers

\begin{equation}
R(f^{*})=P(Y \ne f^{*}(X))=P(Y=1)P(Y=0|X=x)+P(Y=0)P(Y=1|X=x)
\end{equation}

#### example: rare disease

define class 1 = "disease", 0 = "disease-free". $$\pi_1=1\%, \pi_0=99\%$$. 

recall the Bayes rule $$f(x)=\mathcal{1}(P(Y=1|X=x) > P(Y=0|X=x))$$

- posterioe class probability P(Y=j|X=x) gives updated probabilities after observing x
- if $$P(Y=1|X=x) > 0.5$$, thenwe randomly assign data to one class.

Example: Assume a certain rare disease occurs among 1% of the
population. There is a test for this disease: 99.5% of the disease
will test positive, and only 0.5% of the disease-free group will test
positive. (We assume the false positive and false negative rate are
both 0.005.) Now a person comes with a positive test result.
What is the prediction rule?

the conditional probability of X given Y is 

$$
P(X=+|Y=1)=0.995, P(X=-|Y=0)=0.005 \\
P(X=+|Y=0)=0.005, P(X=-|Y=0)=0.995
$$

using Bayes' Theorem,

$$
P(Y=1|X=+) = \frac{P(X=+|Y=1)P(Y=1)}{P(X=+)} = \frac{P(X=+|Y=1)P(Y=1)}{P(X=+|Y=1)P(Y=1)+P(X=+|Y=0)P(Y=0)} = \frac{0.995*0.01}{0.995*0.01+0.005*0.99} = 0.668 \\
P(Y=0|X=+) = \frac{P(X=+|Y=0)P(Y=0)}{P(X=+)} = \frac{0.005*0.99}{0.995*0.01+0.005*0.99} = 0.332
$$

Since P(Y = 0|X = +) = 0.332 < 0.668, the Bayes rule assigns a person with the “+” test result to class “disease”.

Similarly, P(Y = 0|X = −) = 0.9999, P(Y = 1|X = −) = 0.0001, so the Bayes rule assigns a person with the “-” test result to class “disease-free”.


#### Unequal cost
the Bayes rule under unequal cost is given by

\begin{equation}
f^{*}(x)=
\begin{cases} 
1 \quad if \quad C(1,0)P(Y=1|X=x) > C(0,1)P(Y=0|X=x) \\
0 \quad if \quad C(1,0)P(Y=1|X=x) < C(0,1)P(Y=0|X=x)
\end{cases}
\end{equation}
### Linear classification methods
two popular linear classifiers

- Linear Discriminant Analysis (LDA)
- Logistic Regression models (LR)
  - both models rely on the linear-odd assumption, indirectly or directly.
  - LDA and LR estimate the coefficients in a different ways.


linear-logit model (LDA and LR): assume that the logit is linear in x:

\begin{equation}
\log \frac{P(Y=1|X=x)}{P(Y=0|X=x)}=w^Tx+b
\end{equation}

posterior probability:

\begin{equation}
P(Y=1|X=x)=\frac{e^{w^Tx+b}}{1+e^{w^Tx+b}}=\frac{1}{1+e^{-w^Tx-b}} \\
P(Y=0|X=x)=\frac{1}{1+e^{w^Tx+b}}
\end{equation}

under equal-cost, the decision boundary is given by $$\{x|w^Tx+b=0\}=\{x|P(Y=1|X=x)=0\}$$.

#### LDA

LDA assumptions 

- each class density is multivariate Guassian: $$X|Y_j \sim N(\mu_j, \sigma_j)$$
- Equal covariance matrices for each class: $$\sigma_j = \sigma$$

under mixture Guassian assumption, the log-odds is expressed as:

\begin{equation}
  \log \frac{P(Y=1|X=x)}{P(Y=0|X=x)}=\log \frac{\pi_1 \phi(x|\mu_1, \Sigma)/m(x)}{\pi_0 \phi(x|\mu_0, \Sigma)/m(x)} \\
  = \log \frac{\pi_1}{\pi_0} + \log \phi(x|\mu_1, \Sigma)- \log \phi(x|\mu_0, \Sigma) \\
  = \log \frac{\pi_1}{\pi_0} - \frac{1}{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1) + \frac{1}{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0) \\
  = \log \frac{\pi_1}{\pi_0} - \frac{1}{2}(\mu_1+\mu_0)^T\Sigma^{-1}(\mu_1-\mu_0) + x^T\Sigma^{-1}(\mu_1-\mu_0) \\
  \log \frac{\pi_1}{\pi_0} - \frac{1}{2}(\mu_1+\mu_0)^T\beta_1+x^T\beta_1
\end{equation}

under 0-1 loss, the Bayes rule is: assign 1 to x if and only if:

$$
\log \frac{P(Y=1|X=x)}{Y=0|X=x}>0
$$

which is equivalent to: assign 1 to x if and only if:

$$
\bigg [ \log \frac{\pi_1}{\pi_0} - \frac{1}{2}(\mu_1+\mu_0)^T\Sigma^{-1}(\mu_1-\mu_0) \bigg ] + x^T\Sigma^{-1}(\mu_1-\mu_0) > 0.
$$

#### LR

- denote $$\mu=E(Y|X)=P(Y=1|X)$$.
- assuming $$g(\mu)=\log \frac{\mu}{1-\mu}=\beta_0+\beta_1^TX$$.

#### high-dimensional classifiers

- LDA-type 
  - Naive Bayes, Nearest Shrunken Centroid (NSC)
  - sparse LDA, regularized LDA
- penalized logistic regression
- large-margin methods
  - support vector machines (SVM)
- classification tree
  - decision tree
  - random forest
- boosting
### Nonlinear classifier

- KNN
- kernal SVM
- trees, random forests
- ...

#### K-Nearest Neighbor (KNN) classifiers

- pros
  - geenrally low bias
  - no stringent assumptions about data
  - robust against outliers
  - works well for large n small d
- cons
  - potentially high variance
  - work bad for large d small n
  - accuracy severely degraded by noisy or irrelevant features, or the feature scales are not consistent with their importance
    - Much research effort on selecting or scaling features to improve classification, such as optimize feature scaling.


#### kernal SVM
