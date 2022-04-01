# Heston Stochastic Correlation Model (HSCM)
***Numerical option pricing under the Heston model including stochastic correlation*** 

MSc in Finance programme, thesis project at University of Liechtenstein (2020) under the supervision of Dr. Norbert Hilber, Centre for Statistics & Quantitative Finance (IWA) at the Zurich University of Applied Sciences ZHAW. 

## Motivation
A fundamental task in the area of quantitative finance is the accurate modelling of financial markets and capturing its dynamics. Calibrating on market data allows the model to get a sense of the current market conditions, an inevitable prerequisite since less liquid contracts depend on the prices determined by these models. The standard Heston model allows for stochastic volatility modelling but assumes a constant correlation between the return process and its variance process. However, it is unable to generate enough skewness for short-maturity options. 

## Objective
This work studies an extension of the Heston model, proposed by Teng, Ehrhardt and Günther - On the Heston model with stochastic correlation (2016), under which the correlation follows a mean-reverting Ornstein-Uhlenbeck process (HSCM). Modelling the correlation rather stochastically is assumed to allow for a better control of the implied volatility skew and hence, is expected to estimate short-maturity options more accurately. Unfortunately, the analytcal representation of the characteristic function of the HSCM exhibits an error and is therefore not implementable as published by the authors. However, by approximating the corresponding system of ordinary differential equations, the characteristic function is obtained numerically instead, which allows to apply Fourier-Transformation method to compute the option prices. 

## Calibration and Data
The absence of an analytical characteristic function makes the parameter estimation process a computationally tedious task. As a consequence, the calibration is only undertaken for a small set of index option prices, containing only 63 market observations including five maturities. The optimisation problem is solved using the local optimiser SLSQP with arbitrary initial values. 

## Results
Contrary to the expectations and the results within (Teng, Ehrhardt et al., 2016c), the extended Heston model under these conditions does not yield on average a better fit to market data as compared to the pure Heston model. Numerical results however support the better fit for short-maturity options with pricing errors considerably lower compared to the pure Heston model. It is nonetheless expected that the extended Heston model is on average more accurate if calibrated on larger data sets and, since this model is governed by 9 parameters in total, global optimisation algorithms are required for a thorough analysis.
