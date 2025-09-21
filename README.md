This project investigates Bayesian Neural Networks (BNNs), which enhance traditional neural networks by representing weights and biases as probability distributions, enabling uncertainty quantification. Since exact posterior computation in BNNs is intractable, two approximation techniques were compared: Variational Inference (VI) and Markov Chain Monte Carlo (MCMC).

The study used the Breast Cancer Wisconsin dataset, a binary classification task with 569 samples and 30 features. A Bayesian feedforward neural network was implemented in PyTorch with Pyro, using Normal priors, ReLU activations, and Batch Normalization. MCMC was run with the No-U-Turn Sampler (NUTS), while VI used Stochastic Variational Inference (SVI) with ELBO optimisation.

Results showed strong differences in performance:
- MCMC achieved near-perfect classification (accuracy & F1-score of 0.99), producing well-calibrated posteriors and reliable uncertainty estimates, but required ~1.5 hours to run.
- VI was far faster (~4 minutes) but less accurate (ensemble accuracy 0.74, F1-score 0.84) and exhibited greater predictive uncertainty.

The conclusion for this project was that MCMC provides superior posterior estimation and uncertainty calibration, but at a higher computational cost. VI, while faster and more scalable, sacrifices accuracy and reliability. This highlights the trade-off between computational efficiency and predictive robustness in Bayesian inference for Neural Networks.
