"""
Forensic Accounting Module
Implements Altman Z-Score for bankruptcy risk and Beneish M-Score for earnings manipulation detection.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ForensicAnalyzer:
    """
    Analyzes financial statements for signs of distress or manipulation.
    """
    
    def calculate_altman_z_score(self, balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame, market_cap: float) -> Dict[str, Any]:
        """
        Calculate Altman Z-Score for manufacturing/non-manufacturing firms.
        Z = 1.2X1 + 1.4X2 + 3.3X3 + 0.6X4 + 1.0X5
        
        Where:
        X1 = (Current Assets - Current Liabilities) / Total Assets
        X2 = Retained Earnings / Total Assets
        X3 = EBIT / Total Assets
        X4 = Market Value of Equity / Total Liabilities
        X5 = Sales / Total Assets
        """
        try:
            # Extract latest data (column 0 is usually most recent)
            bs = balance_sheet.iloc[:, 0]
            inc = income_stmt.iloc[:, 0]
            
            # Helper to safely get value
            def get_val(series, keys, default=0.0):
                for key in keys:
                    if key in series:
                        return float(series[key])
                return default

            total_assets = get_val(bs, ['Total Assets', 'Assets'])
            current_assets = get_val(bs, ['Current Assets', 'Total Current Assets'])
            current_liab = get_val(bs, ['Current Liabilities', 'Total Current Liabilities'])
            total_liab = get_val(bs, ['Total Liabilities Net Minority Interest', 'Total Liabilities'])
            retained_earnings = get_val(bs, ['Retained Earnings', 'Retained Earnings (Accumulated Deficit)'])
            ebit = get_val(inc, ['EBIT', 'Operating Income', 'Pretax Income']) # Fallback if EBIT missing
            total_revenue = get_val(inc, ['Total Revenue', 'Operating Revenue'])
            
            if total_assets == 0 or total_liab == 0:
                return {"score": 0, "zone": "Unknown", "details": "Missing Total Assets or Liabilities"}
            
            # Calculate Components
            x1 = (current_assets - current_liab) / total_assets
            x2 = retained_earnings / total_assets
            x3 = ebit / total_assets
            x4 = market_cap / total_liab
            x5 = total_revenue / total_assets
            
            # Formula (Original for Public Manufacturing)
            z_score = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
            
            # Interpretation
            if z_score > 2.99:
                zone = "Safe"
            elif z_score > 1.81:
                zone = "Grey"
            else:
                zone = "Distress"
                
            return {
                "score": round(z_score, 2),
                "zone": zone,
                "components": {
                    "liquidity": round(x1, 2),
                    "accumulated_earnings": round(x2, 2),
                    "profitability": round(x3, 2),
                    "leverage": round(x4, 2),
                    "efficiency": round(x5, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Altman Z-Score calculation failed: {e}")
            return {"error": str(e)}

    def calculate_beneish_m_score(self, balance_sheet: pd.DataFrame, income_stmt: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Beneish M-Score to detect earnings manipulation.
        Requires 2 years of data.
        """
        try:
            if balance_sheet.shape[1] < 2 or income_stmt.shape[1] < 2:
                return {"error": "Insufficient historical data (need at least 2 years)"}
            
            # Year t (Current) and Year t-1 (Previous)
            bs_t = balance_sheet.iloc[:, 0]
            bs_t_1 = balance_sheet.iloc[:, 1]
            inc_t = income_stmt.iloc[:, 0]
            inc_t_1 = income_stmt.iloc[:, 1]
            
            def get_val(series, keys, default=0.0):
                for key in keys:
                    if key in series:
                        val = series[key]
                        return float(val) if not pd.isna(val) else default
                return default
            
            # --- Extract Metrics ---
            # Receivables
            rec_t = get_val(bs_t, ['Receivables', 'Net Receivables', 'Accounts Receivable'])
            rec_t_1 = get_val(bs_t_1, ['Receivables', 'Net Receivables', 'Accounts Receivable'])
            
            # Sales
            sales_t = get_val(inc_t, ['Total Revenue', 'Operating Revenue'])
            sales_t_1 = get_val(inc_t_1, ['Total Revenue', 'Operating Revenue'])
            
            # Cost of Goods Sold
            cogs_t = get_val(inc_t, ['Cost Of Revenue', 'Cost of Goods Sold'])
            cogs_t_1 = get_val(inc_t_1, ['Cost Of Revenue', 'Cost of Goods Sold'])
            
            # Assets
            assets_t = get_val(bs_t, ['Total Assets'])
            assets_t_1 = get_val(bs_t_1, ['Total Assets'])
            
            # PPE
            ppe_t = get_val(bs_t, ['Net PPE', 'Property Plant And Equipment Net'])
            ppe_t_1 = get_val(bs_t_1, ['Net PPE', 'Property Plant And Equipment Net'])
            
            # Securities (for AQI) - approximation using Current Assets - Cash - Receivables - Inventory usually
            # Simplified: Non-Current Assets - PPE
            non_curr_assets_t = assets_t - get_val(bs_t, ['Total Current Assets'])
            non_curr_assets_t_1 = assets_t_1 - get_val(bs_t_1, ['Total Current Assets'])
            
            # Depreciation
            # Often handled in Cash Flow, but usually approximated from IS if needed or provided directly
            dep_t = get_val(inc_t, ['Depreciation', 'Depreciation And Amortization']) # NOTE: might be in CF statement
            dep_t_1 = get_val(inc_t_1, ['Depreciation', 'Depreciation And Amortization'])
            
            # SG&A
            sga_t = get_val(inc_t, ['Selling General And Administration'])
            sga_t_1 = get_val(inc_t_1, ['Selling General And Administration'])
            
            # Net Income & CFO for Accruals (TATA)
            ni_t = get_val(inc_t, ['Net Income', 'Net Income Common Stockholders'])
            # Assuming cash_flow statement is not passed here, we might skip TATA or approximate
            # For robustness, we will omit TATA if CF not available, or use (NI - Cash from Ops) if we had CF.
            # Here we will use a simplified model excluding TATA if data is complex, or basic approximation.
            # Standard M-Score uses 8 variables. We will compute 5 key indices which are most robust.
            
            # --- Indices Calculation ---
            
            # 1. DSRI (Days Sales in Receivables Index)
            # (Rec_t / Sales_t) / (Rec_t-1 / Sales_t-1)
            dsri = (rec_t / sales_t) / (rec_t_1 / sales_t_1) if sales_t > 0 and sales_t_1 > 0 and rec_t_1 > 0 else 1.0
            
            # 2. GMI (Gross Margin Index) - Note: (Margin_t-1 / Margin_t)
            # Margin = (Sales - COGS) / Sales
            gm_t = (sales_t - cogs_t) / sales_t if sales_t > 0 else 0
            gm_t_1 = (sales_t_1 - cogs_t_1) / sales_t_1 if sales_t_1 > 0 else 0
            gmi = gm_t_1 / gm_t if gm_t > 0 else 1.0
            
            # 3. AQI (Asset Quality Index)
            # AQ = (1 - (Current Assets + Net PPE + Securities) / Total Assets)
            # Simplified: (Non-Current Assets - PPE) / Total Assets
            aq_t = (non_curr_assets_t - ppe_t) / assets_t if assets_t > 0 else 0
            aq_t_1 = (non_curr_assets_t_1 - ppe_t_1) / assets_t_1 if assets_t_1 > 0 else 0
            aqi = aq_t / aq_t_1 if aq_t_1 > 0 else 1.0
            
            # 4. SGI (Sales Growth Index)
            sgi = sales_t / sales_t_1 if sales_t_1 > 0 else 1.0
            
            # 5. SGAI (SG&A Index)
            # (SGA_t / Sales_t) / (SGA_t-1 / Sales_t-1)
            sgai = (sga_t / sales_t) / (sga_t_1 / sales_t_1) if sales_t > 0 and sales_t_1 > 0 and sga_t_1 > 0 else 1.0
            
            # M-Score Formula (5 Variable Version is common when depreciation/leverage data is noisy)
            # M = -6.065 + 0.823*DSRI + 0.906*GMI + 0.593*AQI + 0.717*SGI + 0.107*SGAI
            
            m_score = -6.065 + (0.823 * dsri) + (0.906 * gmi) + (0.593 * aqi) + (0.717 * sgi) + (0.107 * sgai)
            
            # Interpretation
            # M > -2.22 suggests manipulation
            manipulation_prob = "High" if m_score > -2.22 else "Low"
            
            return {
                "score": round(m_score, 2),
                "probability": manipulation_prob,
                "components": {
                    "dsri": round(dsri, 2),
                    "gmi": round(gmi, 2),
                    "aqi": round(aqi, 2),
                    "sgi": round(sgi, 2),
                    "sgai": round(sgai, 2)
                }
            }

        except Exception as e:
            logger.error(f"Beneish M-Score calculation failed: {e}")
            return {"error": str(e)}

    def get_forensic_report(self, ticker_obj) -> Dict[str, Any]:
        """Generate comprehensive forensic signal"""
        try:
            bs = ticker_obj.balance_sheet
            incomestmt = ticker_obj.income_stmt
            info = ticker_obj.info
            market_cap = info.get('marketCap', 0)
            
            if bs.empty or incomestmt.empty:
                return {"error": "Financial statements not available"}
                
            z_score = self.calculate_altman_z_score(bs, incomestmt, market_cap)
            m_score = self.calculate_beneish_m_score(bs, incomestmt)
            
            return {
                "altman_z_score": z_score,
                "beneish_m_score": m_score
            }
        except Exception as e:
            logger.error(f"Forensic report generation failed: {e}")
            return {"error": str(e)}

_forensic_analyzer_instance = None

def get_forensic_analyzer() -> ForensicAnalyzer:
    global _forensic_analyzer_instance
    if _forensic_analyzer_instance is None:
        _forensic_analyzer_instance = ForensicAnalyzer()
    return _forensic_analyzer_instance
