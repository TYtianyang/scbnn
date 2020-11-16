![](/pic/ivs.png)
:--:
*SC-BNN predicts (20%,80%) quantiles on SPX500 options 2006/01/18*

# Implied Volatility Surface with SC-BNN

The Python code for implied volatility surface project; Source of shape-constrained bayesian neural network.

## Introduction
This is a project lead by Prof. Dacheng Xiu in the Booth School of the University of Chicago. It aims to employ advanced machine learning techniques on the implied volatility surface problem in the field of financial engineering. Along the experiments of benchmark methods including splines, local kernel regression etc., we brought up a new method called shape-constrained bayesian neural network (SC-BNN) to address the challenge of bayesian inference in a constrained parameter space in neural network setting. 

## SC-BNN
The method considers the constrains by modifying the original prior distributions. It introduce the penalizing factor on the original prior to measure how the model satisfies the constrains. In our work, we utilized the probability density function (PDF) of skewed normal distribution with cosine approximation technique as the penalizing factor. On the base of that, we proceed on a simple yet efficient algorithm called "bayes-by-backprop" to do variational inference for the posterior. The method achieves comparable mean prediction to the best of the benchmark methods, and satisfactory quantiles prediction. Most importantly, the method empirically ensures all the model samples satisfying the shape constrains.

## Contribution
1) The work introduced a pioneer yet valid solution to the bayesian inference problem on the implied volatility surface.
2) The work brought up a method that can be utilized in general machine learning topics whenever bayesian inference with shape constrains is called.

## Paper
The work is under preparation to be submitted to IJCNN 2021.

