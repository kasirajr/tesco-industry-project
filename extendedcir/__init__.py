import numpy as np
from scipy.interpolate import CubicSpline


class ExtendedCIR:
    def __init__(self, termStructure, k, theta, sigma, x0):
        self.termStructure = termStructure
        self.k = k
        self.theta = theta
        self.sigma = sigma
        self.x0 = x0
        self.h = np.sqrt(np.square(self.k) + 2 * np.square(self.sigma))

    def fcir(self, t):
        denominator = 2 * self.h + (self.k + self.h) * (np.exp(t * self.h) - 1)
        t1 = (2 * self.k * self.theta * (np.exp(t * self.h) - 1)) / denominator
        t2 = (self.x0 * 4 * np.square(self.h) * np.exp(t * self.h)) / (np.square(denominator))
        return t1 + t2


class InterestRateCIRPP(ExtendedCIR):
    pass


class CreditRiskCIRPP(ExtendedCIR):
    pass


class SSRDSimulation:
    def __init__(self, interestRateModel: InterestRateCIRPP, creditRiskModel: CreditRiskCIRPP, rho: float, N: int,
                 dt: float, T: int):
        self.interestRateModel = interestRateModel
        self.creditRiskModel = creditRiskModel
        self.rho = rho
        self.N = N  # Number of Simulations
        self.dt = dt  # Small Time step
        self.T = T  # Maturity
        self.M = int(T / dt)  # total time steps

    def simulateMC(self):
        i = 1
        interestTermStructureSpline = CubicSpline(self.interestRateModel.termStructure['months'],
                                                  self.interestRateModel.termStructure['rate'])
        xt = np.zeros((self.N, self.M + 1))
        xt[:, 0] = self.interestRateModel.x0
        rt = np.zeros((self.N, self.M + 1))
        while i <= self.M:
            dw1 = np.random.standard_normal(self.N) * np.sqrt(self.dt)
            # for credit
            # dw2 = np.random.standard_normal(self.N) * np.sqrt(self.dt)
            # dz = self.rho * dw1 + np.sqrt(1-np.square(self.rho))*dw2 # Cholesky decomposition
            first_term = np.square(self.interestRateModel.sigma) * np.square(dw1)
            second_term = 4 * (xt[:, i - 1] + (self.interestRateModel.k * self.interestRateModel.theta
                                               - np.square(self.interestRateModel.sigma) / 2) * self.dt
                               * (1 + self.interestRateModel.k * self.dt))
            denominator = 2 * (1 + self.interestRateModel.k * self.dt)
            # sqrt_term = np.sqrt(np.max(first_term + second_term, 0))
            sqrt_term = np.sqrt(np.abs(first_term + second_term))
            rt[:, i - 1] = xt[:, i - 1] + interestTermStructureSpline(self.dt * (i - 1)) - self.interestRateModel.fcir(
                (i - 1) * self.dt)
            xt[:, i] = np.square((self.interestRateModel.sigma * dw1 + sqrt_term) / denominator)
            i += 1
        rt[:, i - 1] = xt[:, i - 1] + interestTermStructureSpline(self.dt * (i - 1)) - self.interestRateModel.fcir(
            (i - 1) * self.dt)
        return rt

# %%
