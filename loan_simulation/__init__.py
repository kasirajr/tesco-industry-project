# Run Loan Book Simulation python code
import math
import numpy as np
import pandas as pd


class LoanBookSimulation:
    def __init__(self, max_months, initial_savings, regulatory_capital_ratio):
        self.max_month = max_months
        self.initial_savings = initial_savings
        self.current_savings = initial_savings
        self.regulatory_capital_ratio = regulatory_capital_ratio
        self.loan_book = []
        self.customer_payments = []
        self.deposit_account = []
        self.prepayed_loans = 0

    # Parameters
    # max_month = 60
    # months = range(max_month)
    # initial_savings = 10 * 1e6  # £1 million
    # current_savings = initial_savings
    # prepayed_loans = 0
    # regulatory_capital_ratio = 0.5
    # loan_book = []
    # customer_payments = []
    # deposit_account = []

    # Market rates simulation using CIR process
    def CIR_process(self, r0, kappa, theta, sigma, dt, n):
        r = np.zeros(n)
        r[0] = r0
        for i in range(1, n):
            gamma = sigma * np.sqrt(math.fabs(r[i - 1]))
            z = np.random.randn()
            r[i] = np.abs(r[i - 1] + kappa * (theta - r[i - 1]) * dt + gamma * np.sqrt(dt) * z)
        return r

    # Customer arrival simulation as a Poisson process
    def simulate_customers(self, lam, duration):
        t = 0
        customers = []
        while t < duration:
            interarrival_time = np.random.exponential(1 / lam)
            t += interarrival_time
            if t < duration:
                customers.append(t)
        return customers

    # Generate loan parameters for a customer
    def generate_loan_parameters(self):
        credit_quality = np.random.uniform(0, 1)
        loan_amount = np.random.randint(1000, 25001)
        loan_term = np.random.randint(12, self.max_month + 1)
        fixed_rate = np.random.uniform(0.02, 0.08)
        return credit_quality, loan_amount, loan_term, fixed_rate

    # Simulate loan book over 60 months
    def simulate_loan_book(self):
        cir_params = {'r0': 0.03, 'kappa': 1, 'theta': 0.045, 'sigma': 0.1}
        market_rates = self.CIR_process(cir_params['r0'], cir_params['kappa'], cir_params['theta'], cir_params['sigma'],
                                        1 / 12, self.max_month)
        discount_factors = np.cumprod(1 / (1 + market_rates / 12))
        customer_arrivals = self.simulate_customers(15, self.max_month)  # Assuming average of 10 customers per month

        # Now perform the simulation:
        for month in range(self.max_month):
            # Remove customers who have completed their repayments to term
            deposit_account_maturities = [loan['loan_amount'] for loan in self.loan_book if loan['loan_term'] > month]
            self.current_savings += sum(deposit_account_maturities)
            self.loan_book = [loan for loan in self.loan_book if
                              (loan['loan_term'] > month and self.deposit_account.append)]
            # Add new customers
            for arrival in customer_arrivals:
                if arrival <= month:
                    credit_quality, loan_amount, loan_term, fixed_rate = self.generate_loan_parameters()
                    loan = {'credit_quality': credit_quality, 'loan_amount': loan_amount, 'loan_term': loan_term,
                            'fixed_rate': fixed_rate}
                    if loan['loan_amount'] <= self.current_savings:
                        self.loan_book.append(loan)
                        self.current_savings -= loan['loan_amount']
            # Calculate monthly customer payments and remove customers who prepay
            for loan in self.loan_book:
                market_rate = market_rates[month]
                if loan['fixed_rate'] < market_rate:
                    monthly_payment = loan['loan_amount'] * loan['fixed_rate'] / 12
                else:
                    self.current_savings += loan['loan_amount']
                    self.prepayed_loans += loan['loan_amount']
                    self.loan_book.remove(loan)
                    continue
                loan['loan_amount'] -= monthly_payment
                if loan['loan_amount'] <= 0:
                    self.loan_book.remove(loan)
                self.customer_payments.append(monthly_payment)
            # Save current_savings
            self.deposit_account.append(discount_factors[month] * self.current_savings)

    def loan_book_segmentation(self):
        credit_quality = np.zeros(len(self.loan_book))
        loan_term = np.zeros(len(self.loan_book))
        loan_amount = np.zeros(len(self.loan_book))
        fixed_rate = np.zeros(len(self.loan_book))

        # Extract the values from the loan book
        for i, loan in enumerate(self.loan_book):
            credit_quality[i] = loan['credit_quality']
            loan_term[i] = loan['loan_term']
            loan_amount[i] = loan['loan_amount']
            fixed_rate[i] = loan['fixed_rate']

            # Define cut-off values for credit quality, loan term, loan amount, and fixed rate
            credit_quality_cutoffs = np.percentile(credit_quality, [33, 66]).round(4)
            loan_term_cutoffs = np.percentile(loan_term, [33, 66]).round(0)
            loan_amount_cutoffs = np.percentile(loan_amount, [33, 66]).round(-2)  # nearest 100
            fixed_rate_cutoffs = np.percentile(fixed_rate, [33, 66]).round(4)

            # Segment the personal loans book
            credit_quality_segment = np.digitize(credit_quality, credit_quality_cutoffs)
            loan_term_segment = np.digitize(loan_term, loan_term_cutoffs)
            loan_amount_segment = np.digitize(loan_amount, loan_amount_cutoffs)
            fixed_rate_segment = np.digitize(fixed_rate, fixed_rate_cutoffs)

            # Make dataframe of book with segments
            loan_book_segmentation = pd.DataFrame({
                'credit_quality': credit_quality.round(4),
                'credit_quality_segment': credit_quality_segment,
                'loan_term': loan_term.round(0),
                'loan_term_segment': loan_term_segment,
                'loan_amount': loan_amount.round(-2),
                'loan_amount_segment': loan_amount_segment,
                'fixed_rate': fixed_rate.round(4),
                'fixed_rate_segment': fixed_rate_segment})

            pd.set_option('display.max_columns', len(loan_book_segmentation.columns))
        return loan_book_segmentation

    def split_loan_into_tranches(self, loans):
        loan_tranches = {}
        types = [0, 1, 2]
        dict_map = {'0': 'L', '1': 'M', '2': 'H'}
        for amount_segment in types:
            for term_segment in types:
                for credit_segment in types:
                    for fixed_rate_segment in types:
                        seg_key = [str(amount_segment), str(term_segment), str(credit_segment), str(fixed_rate_segment)]
                        segment = ''.join([dict_map[digit] for digit in seg_key])
                        loan_tranches[segment] = loans[(loans['loan_amount_segment'] == amount_segment) & (loans[
                                                                                                               'loan_term_segment'] == term_segment) &
                                                       (loans['credit_quality_segment'] == credit_segment) & (loans[
                                                                                                                  'fixed_rate_segment'] == fixed_rate_segment)]

        return loan_tranches
