"""
Package Import
"""
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp
import warnings
import argparse
import sys

"""
Project Setup
"""
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    """
    NOTE: You can modify the initialization function
    """

    def __init__(self, price, exclude, lookback=150, gamma=3):
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        self.lookback = lookback
        self.gamma = gamma

    def calculate_weights(self):
        # Get the assets by excluding the specified column
        assets = self.price.columns[self.price.columns != self.exclude]

        # Calculate the portfolio weights
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        )

        """
        TODO: Complete Task 4 Below
        """
        
        def mv_opt(R_n_full, R_n_mu, gamma):
            Sigma_history = R_n_full.cov().values
            mu = R_n_mu.mean().values
            n = len(R_n_full.columns)
            
            avg_variance = np.diag(Sigma_history).mean()
            Sigma_target = np.identity(n) * avg_variance
            Sigma = 0.9 * Sigma_history + 0.1 * Sigma_target

            fallback_solution = [1/n] * n 

            try:
                with gp.Env(empty=True) as env:
                    env.setParam("OutputFlag", 0)
                    env.setParam("DualReductions", 0)
                    env.start()
                    with gp.Model(env=env, name="portfolio_mv_opt_sharpe") as model:
                        
                        w = model.addMVar(n, name="w", lb=-0.1, ub=0.3)

                        front = 0
                        for i in range(n):
                            front += w[i] * mu[i]

                        back = 0
                        for i in range(n):
                            for j in range(n):
                                back += w[i] * Sigma[i, j] * w[j]
                        
                        model.setObjective(front - ((gamma / 2) * back), gp.GRB.MAXIMIZE)
                        model.addConstr(w.sum() == 0.9)

                        model.optimize()

                        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.SUBOPTIMAL:
                            solution = []
                            for i in range(n):
                                var = model.getVarByName(f"w[{i}]")
                                # print(f"w {i} = {var.X}")
                                solution.append(var.X)
                            return solution
                        else:
                            return fallback_solution
            except Exception as e:
                return fallback_solution
        
        for i in range(self.lookback + 1, len(self.price)):
            R_n_full = self.returns[assets].iloc[i - self.lookback : i]
            R_n_mu = self.returns[assets].iloc[i - 120 : i]
            
            self.portfolio_weights.loc[self.price.index[i], assets] = mv_opt(
                R_n_full, R_n_mu, self.gamma, 
            )
        
        """
        TODO: Complete Task 4 Above
        """

        self.portfolio_weights.ffill(inplace=True)
        self.portfolio_weights.fillna(0, inplace=True)

    def calculate_portfolio_returns(self):
        # Ensure weights are calculated
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # Calculate the portfolio returns
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # Ensure portfolio returns are calculated
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)
