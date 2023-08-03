import numpy as np
from scipy.interpolate import CubicSpline
import cdstools

class ExtendedCIR:
    def __init__(self, k, theta, sigma, x0):
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

    def A(self, t, T):
        tau = T - t
        power = 2 * self.k * self.theta / self.sigma ** 2
        denominator = 2 * self.h + (self.k + self.h) * (np.exp(tau * self.h) - 1)
        numerator = 2 * self.h * (np.exp((self.k + self.h) * tau / 2))
        return (numerator / denominator) ** power

    def B(self, t, T):
        tau = T - t
        numerator = 2 * (np.exp(tau * self.h) - 1)
        denominator = 2 * self.h + (self.k + self.h) * (np.exp(tau * self.h) - 1)
        return numerator / denominator

    def price(self, t, T, xt):
        pT = (1 / ((1 + self.termStructureSpline(T)) ** T)) * self.A(0, t) * np.exp(-self.B(0, t) * self.x0)
        pt = (1 / ((1 + self.termStructureSpline(t)) ** t)) * self.A(0, T) * np.exp(-self.B(0, T) * self.x0)
        return (pT / pt) * self.pcir(t, T, xt)

    def pcir(self, t, T, xt):
        return self.A(t, T) * np.exp(-self.B(t, T) * xt)


class InterestRateCIRPP(ExtendedCIR):
    def __init__(self, termStructure, k, theta, sigma, x0):
        super().__init__(k, theta, sigma, x0)
        self.termStructureSpline = CubicSpline(termStructure['years'],
                                               termStructure['rate'])


class CreditRiskCIRPP(ExtendedCIR):
    def __init__(self, interestTermStructure, creditTermStructure, k, theta, sigma, x0, rr, pf):
        super().__init__(k, theta, sigma, x0)
        prob = cdstools.CDS_bootstrap(creditTermStructure['spread'],
                                      interestTermStructure['rate'],
                                      creditTermStructure['years'],
                                      interestTermStructure['years'], pf, rr)

        self.termStructureSpline = CubicSpline(creditTermStructure['years'], prob[0])

    def fcir(self, t):
        return -np.log(self.pcir(0, t, self.x0))


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
        xt = np.zeros((self.N, self.M + 1))
        xt[:, 0] = self.interestRateModel.x0
        yt = np.zeros((self.N, self.M + 1))
        yt[:, 0] = self.creditRiskModel.x0
        rt = np.zeros((self.N, self.M + 1))
        ct = np.zeros((self.N, self.M + 1))
        while i <= self.M:
            dw1 = np.random.standard_normal(self.N) * np.sqrt(self.dt)
            dw2 = np.random.standard_normal(self.N) * np.sqrt(self.dt)
            dz = self.rho * dw1 + np.sqrt(1 - np.square(self.rho)) * dw2  # Cholesky decomposition

            rt[:, i - 1] = xt[:, i - 1] + self.interestRateModel.termStructureSpline(
                self.dt * (i - 1)) - self.interestRateModel.fcir(
                (i - 1) * self.dt)
            ct[:, i - 1] = yt[:, i - 1] + self.creditRiskModel.termStructureSpline(
                self.dt * (i - 1)) - self.creditRiskModel.fcir(
                (i - 1) * self.dt)
            xt[:, i] = self.nextTerm(self.interestRateModel, dw1, xt[:, i - 1])
            yt[:, i] = self.nextTerm(self.creditRiskModel, dz, yt[:, i - 1])
            i += 1
        rt[:, i - 1] = xt[:, i - 1] + self.interestRateModel.termStructureSpline(
            self.dt * (i - 1)) - self.interestRateModel.fcir(
            (i - 1) * self.dt)
        ct[:, i - 1] = yt[:, i - 1] + self.creditRiskModel.termStructureSpline(
            self.dt * (i - 1)) - self.creditRiskModel.fcir(
            (i - 1) * self.dt)
        return rt, ct, rt + ct

    def nextTerm(self, model, dw, previousTerm):
        first_term = np.square(model.sigma) * np.square(dw)
        second_term = 4 * (previousTerm + (model.k * model.theta
                                           - np.square(model.sigma) / 2) * self.dt
                           * (1 + model.k * self.dt))
        denominator = 2 * (1 + model.k * self.dt)
        # sqrt_term = np.sqrt(np.max(first_term + second_term, 0))
        sqrt_term = np.sqrt(np.abs(first_term + second_term))
        return np.square((model.sigma * dw + sqrt_term) / denominator)

# %%
