#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Non-parametric confidence tests for calibration errors

"""

# Simulate 1'000 empirical coverages of perfect prediction sets
# (1) window size 25, length 500
np.random.seed(1)
C = [[np.random.binomial(n=1, p=0.9) for j in range(500)] for i in range(10000)]
emp_cov = mov_avg(np.array(C), 25)
MAD = np.mean(abs(emp_cov - 0.9), axis=1)
MAD_mean = 100.*MAD.mean()
MAD_sd = (100.*MAD).std()
print("MAD (mean):", round(MAD_mean, 2), "%")
print("MAD (sd):", round(MAD_sd, 2), "%")
print("MAD (95%):", round(np.quantile(100.*MAD, 0.95), 2))

# (1) window size 50, length 1'000
np.random.seed(1)
C = [[np.random.binomial(n=1, p=0.9) for j in range(1000)] for i in range(10000)]
emp_cov = mov_avg(np.array(C), 50)
MAD = np.mean(abs(emp_cov - 0.9), axis=1)
MAD_mean = 100.*MAD.mean()
MAD_sd = (100.*MAD).std()
print("MAD (mean):", round(MAD_mean, 2), "%")
print("MAD (sd):", round(MAD_sd, 2), "%")
print("MAD (95%):", round(np.quantile(100.*MAD, 0.95), 2))