"""
Valuation Module
Implements DCF (Discounted Cash Flow) and Relative Valuation models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ValuationAnalyzer:
    """
    Calculates intrinsic value using various models.
    """
    
    def calculate_dcf(self, cashflow_stmt: pd.DataFrame, income_stmt: pd.DataFrame, balance_sheet: pd.DataFrame, beta: float = 1.0, risk_free_rate: float = 0.07, market_return: float = 0.12) -> Dict[str, Any]:
        """
        Calculate Intrinsic Value using 2-Stage DCF Model.
        """
        try:
            # Helper
            def get_val(series, keys, default=0.0):
                for key in keys:
                    if key in series:
                        val = series[key]
                        return float(val) if not pd.isna(val) else default
                return default

            # Get latest available full year Free Cash Flow
            # Free Cash Flow = Operating Cash Flow - Capital Expenditure
            cf = cashflow_stmt.iloc[:, 0]
            ocf = get_val(cf, ['Total Cash From Operating Activities', 'Operating Cash Flow'])
            capex = get_val(cf, ['Capital Expenditure', 'Net PPE Purchase And Sale'])
            
            # Using absolute value of Capex as it's usually negative
            fcf = ocf + capex # Check sign convention: usually capex is negative in CF stmt
            
            if fcf <= 0:
                # Fallback: Use Net Income + Deprec if FCF is weird or negative (simplified owner earnings)
                inc = income_stmt.iloc[:, 0]
                ni = get_val(inc, ['Net Income', 'Net Income Common Stockholders'])
                dep = get_val(inc, ['Depreciation', 'Depreciation And Amortization'])
                fcf = ni + dep
            
            if fcf <= 0:
                return {"error": "Negative Free Cash Flow, DCF unreliable"}

            # WACC Calculation (Cost of Equity only for simplicity, assuming largely equity financed for tech/retail or simplified model)
            # Cost of Equity = Rf + Beta * (Rm - Rf)
            # India Rf ~ 7%, Rm ~ 12%
            cost_of_equity = risk_free_rate + beta * (market_return - risk_free_rate)
            wacc = max(0.08, min(cost_of_equity, 0.15)) # Cap between 8% and 15% to be realistic
            
            # Growth Rates
            # Estimate from historicals if available, or conservative placeholder
            # For this version, we assume a conservative 12% growth for next 5 years, then 4% terminal
            growth_stage_1 = 0.12 
            growth_terminal = 0.05
            
            # Projections (5 Years)
            future_fcf = []
            for i in range(1, 6):
                fcf_proj = fcf * ((1 + growth_stage_1) ** i)
                future_fcf.append(fcf_proj)
                
            # Terminal Value
            # TV = (Final FCF * (1 + g)) / (WACC - g)
            terminal_value = (future_fcf[-1] * (1 + growth_terminal)) / (wacc - growth_terminal)
            
            # Discounting
            dcf_value = 0
            for i, val in enumerate(future_fcf):
                dcf_value += val / ((1 + wacc) ** (i + 1))
            
            dcf_value += terminal_value / ((1 + wacc) ** 5)
            
            # Shares Outstanding
            bs = balance_sheet.iloc[:, 0]
            # yfinance sometimes puts shares in info, not balance sheet. 
            # We will return the Total Market Value, let caller divide by shares/price check.
            # But usually we want per share.
            
            # Net Debt Adjustment
            total_debt = get_val(bs, ['Total Debt', 'Total Liabilities Net Minority Interest']) # Rough proxy if debt missing
            cash = get_val(bs, ['Cash And Cash Equivalents', 'Total Cash'])
            net_debt = total_debt - cash
            
            equity_value = dcf_value - net_debt
            
            return {
                "intrinsic_value_total": equity_value,
                "method": "DCF (5Y Growth)",
                "assumptions": {
                    "wacc": round(wacc * 100, 2),
                    "growth_rate": round(growth_stage_1 * 100, 2),
                    "terminal_rate": round(growth_terminal * 100, 2)
                }
            }

        except Exception as e:
            logger.error(f"DCF calculation failed: {e}")
            return {"error": str(e)}

    def calculate_graham_number(self, eps: float, book_value_per_share: float) -> float:
        """
        Calculate Graham Number = Sqrt(22.5 * EPS * BVPS)
        """
        try:
            if eps < 0 or book_value_per_share < 0:
                return 0.0
            return (22.5 * eps * book_value_per_share) ** 0.5
        except:
            return 0.0

    def get_valuation_report(self, ticker_obj) -> Dict[str, Any]:
        """Generate comprehensive valuation"""
        try:
            info = ticker_obj.info
            
            # Basic info
            current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            shares = info.get('sharesOutstanding', 0)
            eps = info.get('trailingEps', 0)
            bvps = info.get('bookValue', 0)
            beta = info.get('beta', 1.0)
            
            if shares == 0:
                return {"error": "Shares outstanding missing"}

            # 1. Graham Number
            graham_num = self.calculate_graham_number(eps, bvps)
            
            # 2. DCF
            dcf_res = self.calculate_dcf(
                ticker_obj.cashflow, 
                ticker_obj.income_stmt, 
                ticker_obj.balance_sheet,
                beta=beta if beta else 1.0
            )
            
            dcf_price = 0.0
            if 'intrinsic_value_total' in dcf_res:
                dcf_price = dcf_res['intrinsic_value_total'] / shares
                
            # Assessments
            signals = []
            if dcf_price > 0:
                margin_of_safety = (dcf_price - current_price) / dcf_price
                if margin_of_safety > 0.3:
                    signals.append("Undervalued (DCF)")
                elif margin_of_safety < -0.3:
                    signals.append("Overvalued (DCF)")
            
            if graham_num > 0:
                if current_price < graham_num * 0.8:
                    signals.append("Undervalued (Graham)")
            
            return {
                "current_price": current_price,
                "fair_value_dcf": round(dcf_price, 2),
                "fair_value_graham": round(graham_num, 2),
                "signals": signals,
                "dcf_details": dcf_res
            }
            
        except Exception as e:
            logger.error(f"Valuation report failed: {e}")
            return {"error": str(e)}

_valuation_analyzer_instance = None

def get_valuation_analyzer() -> ValuationAnalyzer:
    global _valuation_analyzer_instance
    if _valuation_analyzer_instance is None:
        _valuation_analyzer_instance = ValuationAnalyzer()
    return _valuation_analyzer_instance
