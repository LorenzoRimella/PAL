# Consistent and fast inference in compartmental models of epidemics using Poisson Approximate Likelihoods

## Abstract
Addressing the challenge of scaling-up epidemiological inference to complex and het-
erogeneous models, we introduce Poisson Approximate Likelihood (PAL) methods. In contrast to
the popular ODE approach to compartmental modelling, in which a large population limit is used
to motivate a deterministic model, PALs are derived from approximate filtering equations for finite-
population, stochastic compartmental models, and the large population limit drives consistency of
maximum PAL estimators. Our theoretical results appear to be the first likelihood-based param-
eter estimation consistency results which apply to a broad class of partially observed stochastic
compartmental models and address the large population limit. PALs are simple to implement,
involving only elementary arithmetic operations and no tuning parameters, and fast to evaluate,
requiring no simulation from the model and having computational cost independent of population
size. Through examples we demonstrate how PALs can be used to: fit an age-structured model of
influenza, taking advantage of automatic differentiation in Stan; compare over-dispersion mecha-
nisms in a model of rotavirus by embedding PALs within sequential Monte Carlo; and evaluate the
role of unit-specific parameters in a meta-population model of measles.

## Contents:
The repository contains two folders: 
- SyntheticDataExperiments;
- RealDataExperiments.

SyntheticDataExperiments collects all the experiments on synthetic data supporting the theoretical results:
- LLN;
- filtering recursion limits;
- convergence of the maximum PAL estimator;
- convergence of the maximum PAL estimator with identifiability issues;
- PALSMC filtering
- convergence of the maximum PALSMC estimator.

RealDataExperiments collects the real data experiments:
- delayed acceptance PMCMC for the British boarding school influenza outbreak in 1978;
- inference on an age-structured through HMC in Stan for the influenza outbreak in Wales in 1957;
- over-dispersed model comparisons and selection - applied to the transmission of Rotavirus in Germany 2001-2008
- inference in a gravity model for measles data (with overdispersion and city specific parameters).
