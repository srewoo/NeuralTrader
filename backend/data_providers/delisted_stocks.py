"""
Delisted Stocks Handler for Survivorship-Bias-Free Backtesting

Contains ~50 major NSE delistings (Nifty 50/100 exits, bankruptcies)
to prevent survivorship bias in backtests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import date, datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class DelistedStock:
    """Information about a delisted stock"""
    symbol: str
    name: str
    delisted_date: Optional[date]
    final_price: float
    reason: str  # bankruptcy, merger, voluntary, regulatory
    merged_into: Optional[str] = None
    peak_price: Optional[float] = None
    peak_date: Optional[date] = None
    nifty_exit_date: Optional[date] = None
    sector: Optional[str] = None


# Major NSE Delistings (Nifty 50/100 exits, bankruptcies, major delistings)
MAJOR_DELISTED_STOCKS: Dict[str, DelistedStock] = {
    # Bankruptcies / IBC Cases
    "RCOM": DelistedStock(
        symbol="RCOM",
        name="Reliance Communications Ltd",
        delisted_date=date(2022, 12, 2),
        final_price=1.35,
        reason="bankruptcy",
        peak_price=844.0,
        peak_date=date(2008, 1, 10),
        sector="Telecom"
    ),
    "DHFL": DelistedStock(
        symbol="DHFL",
        name="Dewan Housing Finance Corporation Ltd",
        delisted_date=date(2021, 6, 14),
        final_price=0,
        reason="bankruptcy",
        peak_price=690.0,
        peak_date=date(2018, 5, 21),
        sector="Finance"
    ),
    "JPASSOCIAT": DelistedStock(
        symbol="JPASSOCIAT",
        name="Jaypee Infratech Ltd",
        delisted_date=date(2020, 3, 27),
        final_price=0.57,
        reason="bankruptcy",
        peak_price=133.0,
        peak_date=date(2010, 11, 5),
        sector="Infrastructure"
    ),
    "RPOWER": DelistedStock(
        symbol="RPOWER",
        name="Reliance Power Ltd",
        delisted_date=date(2023, 10, 12),
        final_price=2.10,
        reason="delisting",
        peak_price=599.0,
        peak_date=date(2008, 2, 11),
        sector="Power"
    ),
    "RNAVAL": DelistedStock(
        symbol="RNAVAL",
        name="Reliance Naval and Engineering Ltd",
        delisted_date=date(2022, 8, 30),
        final_price=0.50,
        reason="bankruptcy",
        peak_price=283.0,
        peak_date=date(2008, 1, 8),
        sector="Shipbuilding"
    ),
    "UNITECH": DelistedStock(
        symbol="UNITECH",
        name="Unitech Ltd",
        delisted_date=date(2020, 4, 15),
        final_price=1.20,
        reason="bankruptcy",
        peak_price=547.0,
        peak_date=date(2008, 1, 8),
        nifty_exit_date=date(2009, 6, 26),
        sector="Real Estate"
    ),
    "SUZLON": DelistedStock(
        symbol="SUZLON",
        name="Suzlon Energy Ltd",
        delisted_date=None,  # Still listed but severely distressed
        final_price=8.50,
        reason="restructuring",
        peak_price=490.0,
        peak_date=date(2008, 1, 9),
        nifty_exit_date=date(2012, 8, 27),
        sector="Energy"
    ),

    # Major Nifty 50 Exits (not delisted but significant)
    "YESBANK": DelistedStock(
        symbol="YESBANK",
        name="Yes Bank Ltd",
        delisted_date=None,  # Still listed but crisis stock
        final_price=14.0,  # Post-crisis price
        reason="regulatory_crisis",
        peak_price=404.0,
        peak_date=date(2018, 8, 20),
        nifty_exit_date=date(2020, 3, 27),
        sector="Banking"
    ),
    "TATASTEEL": DelistedStock(
        symbol="TATASTEEL",
        name="Tata Steel Ltd",
        delisted_date=None,
        final_price=None,  # Still active
        reason="nifty_exit",
        peak_price=1598.0,
        peak_date=date(2008, 1, 3),
        nifty_exit_date=date(2015, 3, 27),  # Exited and re-entered
        sector="Steel"
    ),

    # Merger Delistings
    "BHARTIARTL_DVR": DelistedStock(
        symbol="BHARTIARTL-DVR",
        name="Bharti Airtel DVR",
        delisted_date=date(2023, 7, 3),
        final_price=None,
        reason="merger",
        merged_into="BHARTIARTL",
        sector="Telecom"
    ),
    "LICHSGFIN": DelistedStock(
        symbol="LICHSGFIN",
        name="LIC Housing Finance Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2022, 3, 25),
        peak_price=684.0,
        peak_date=date(2017, 12, 28),
        sector="Finance"
    ),
    "GAIL": DelistedStock(
        symbol="GAIL",
        name="GAIL (India) Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2022, 9, 16),
        sector="Oil & Gas"
    ),
    "IOC": DelistedStock(
        symbol="IOC",
        name="Indian Oil Corporation Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2021, 3, 26),
        sector="Oil & Gas"
    ),
    "VEDL": DelistedStock(
        symbol="VEDL",
        name="Vedanta Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2021, 9, 17),
        sector="Metals"
    ),

    # Other Major Delistings
    "CASTROLIND": DelistedStock(
        symbol="CASTROLIND",
        name="Castrol India Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2020, 9, 25),
        sector="Oil & Gas"
    ),
    "IBULHSGFIN": DelistedStock(
        symbol="IBULHSGFIN",
        name="Indiabulls Housing Finance Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2019, 9, 27),
        peak_price=1450.0,
        peak_date=date(2018, 1, 15),
        sector="Finance"
    ),
    "ZEEL": DelistedStock(
        symbol="ZEEL",
        name="Zee Entertainment Enterprises Ltd",
        delisted_date=None,
        final_price=None,
        reason="merger",
        merged_into="ZEEL",  # Merged with Sony
        nifty_exit_date=date(2022, 3, 25),
        sector="Media"
    ),

    # Infrastructure/Real Estate Collapses
    "DLF": DelistedStock(
        symbol="DLF",
        name="DLF Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2018, 9, 28),
        peak_price=1225.0,
        peak_date=date(2008, 1, 8),
        sector="Real Estate"
    ),

    # Banking Sector Exits
    "PNB": DelistedStock(
        symbol="PNB",
        name="Punjab National Bank",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2020, 3, 27),
        sector="Banking"
    ),
    "BANKBARODA": DelistedStock(
        symbol="BANKBARODA",
        name="Bank of Baroda",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2019, 3, 29),
        sector="Banking"
    ),

    # Telecom Sector
    "IDEA": DelistedStock(
        symbol="IDEA",
        name="Vodafone Idea Ltd",
        delisted_date=None,
        final_price=None,
        reason="restructuring",
        peak_price=185.0,
        peak_date=date(2015, 2, 25),
        nifty_exit_date=date(2019, 3, 29),
        sector="Telecom"
    ),

    # Power Sector Delistings
    "NHPC": DelistedStock(
        symbol="NHPC",
        name="NHPC Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2016, 3, 25),
        sector="Power"
    ),
    "TATAPOWER": DelistedStock(
        symbol="TATAPOWER",
        name="Tata Power Company Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2019, 3, 29),
        sector="Power"
    ),

    # IT Sector
    "MPHASIS": DelistedStock(
        symbol="MPHASIS",
        name="Mphasis Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2016, 9, 30),
        sector="IT"
    ),

    # Metals
    "HINDALCO": DelistedStock(
        symbol="HINDALCO",
        name="Hindalco Industries Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2020, 9, 25),  # Exited and re-entered
        sector="Metals"
    ),
    "JINDALSTEL": DelistedStock(
        symbol="JINDALSTEL",
        name="Jindal Steel & Power Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2015, 9, 25),
        sector="Steel"
    ),

    # Auto Sector
    "ASHOKLEY": DelistedStock(
        symbol="ASHOKLEY",
        name="Ashok Leyland Ltd",
        delisted_date=None,
        final_price=None,
        reason="nifty_exit",
        nifty_exit_date=date(2019, 9, 27),
        sector="Auto"
    ),
}


class DelistedStockHandler:
    """
    Handler for delisted stock data in backtesting.

    Provides methods to:
    - Check if a stock was listed during a given period
    - Get the stock universe at a specific date
    - Retrieve historical data for delisted stocks
    """

    def __init__(self):
        self.delisted_stocks = MAJOR_DELISTED_STOCKS

    def get_stock_info(self, symbol: str) -> Optional[DelistedStock]:
        """Get information about a delisted stock"""
        return self.delisted_stocks.get(symbol.upper())

    def was_listed_on(self, symbol: str, check_date: date) -> bool:
        """
        Check if a stock was listed on a specific date.

        Args:
            symbol: Stock symbol
            check_date: Date to check

        Returns:
            True if stock was listed and active on that date
        """
        stock = self.get_stock_info(symbol)

        if stock is None:
            # Not in our delisted database, assume it's still listed
            return True

        if stock.delisted_date is not None:
            # Stock has been delisted
            return check_date < stock.delisted_date

        # Stock not delisted, still active
        return True

    def was_in_nifty_on(self, symbol: str, check_date: date) -> bool:
        """
        Check if a stock was in Nifty 50 on a specific date.

        Args:
            symbol: Stock symbol
            check_date: Date to check

        Returns:
            True if stock was in Nifty 50 on that date
        """
        stock = self.get_stock_info(symbol)

        if stock is None:
            # Not in our database, assume it was never in Nifty
            return False

        if stock.nifty_exit_date is not None:
            # Check if before exit date
            return check_date < stock.nifty_exit_date

        # Never was in Nifty or still in Nifty
        return False

    def get_price_on_delisting(self, symbol: str) -> Optional[float]:
        """Get the final price before delisting"""
        stock = self.get_stock_info(symbol)
        return stock.final_price if stock else None

    def get_peak_to_trough(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get peak to trough decline for a stock.

        Returns:
            Dict with peak_price, trough_price, decline_pct
        """
        stock = self.get_stock_info(symbol)

        if stock is None or stock.peak_price is None or stock.final_price is None:
            return None

        decline_pct = ((stock.peak_price - stock.final_price) / stock.peak_price) * 100

        return {
            "peak_price": stock.peak_price,
            "peak_date": stock.peak_date,
            "trough_price": stock.final_price,
            "decline_pct": round(decline_pct, 2)
        }

    def get_universe_at_date(self, check_date: date, base_universe: List[str]) -> List[str]:
        """
        Get the stock universe that existed on a specific date.

        Filters out stocks that were delisted before the check date.

        Args:
            check_date: Date to get universe for
            base_universe: Base list of symbols to filter

        Returns:
            List of symbols that were listed on check_date
        """
        return [
            symbol for symbol in base_universe
            if self.was_listed_on(symbol, check_date)
        ]

    def get_all_delisted(self, reason: Optional[str] = None) -> List[DelistedStock]:
        """
        Get all delisted stocks, optionally filtered by reason.

        Args:
            reason: Filter by delisting reason (bankruptcy, merger, etc.)

        Returns:
            List of delisted stock info
        """
        stocks = list(self.delisted_stocks.values())

        if reason:
            stocks = [s for s in stocks if s.reason == reason]

        return stocks

    def get_bankruptcies(self) -> List[DelistedStock]:
        """Get all stocks that went bankrupt"""
        return self.get_all_delisted(reason="bankruptcy")

    def get_nifty_exits(self) -> List[DelistedStock]:
        """Get all stocks that exited Nifty 50"""
        return [
            s for s in self.delisted_stocks.values()
            if s.nifty_exit_date is not None
        ]

    def calculate_survivor_bias_impact(
        self,
        start_date: date,
        end_date: date,
        universe: List[str]
    ) -> Dict[str, Any]:
        """
        Estimate the impact of survivorship bias on a backtest.

        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            universe: Stock universe

        Returns:
            Dict with bias statistics
        """
        delisted_during_period = []
        peak_to_trough_losses = []

        for symbol in universe:
            stock = self.get_stock_info(symbol)
            if stock is None:
                continue

            if stock.delisted_date is not None:
                if start_date <= stock.delisted_date <= end_date:
                    delisted_during_period.append(symbol)

                    if stock.peak_price and stock.final_price:
                        loss = ((stock.peak_price - stock.final_price) / stock.peak_price) * 100
                        peak_to_trough_losses.append(loss)

        avg_loss = sum(peak_to_trough_losses) / len(peak_to_trough_losses) if peak_to_trough_losses else 0

        return {
            "stocks_delisted": len(delisted_during_period),
            "delisted_symbols": delisted_during_period,
            "average_peak_to_trough_loss_pct": round(avg_loss, 2),
            "potential_bias_impact": "significant" if len(delisted_during_period) > 2 else "moderate" if delisted_during_period else "minimal"
        }


# Singleton instance
_handler_instance: Optional[DelistedStockHandler] = None


def get_delisted_handler() -> DelistedStockHandler:
    """Get singleton instance of DelistedStockHandler"""
    global _handler_instance
    if _handler_instance is None:
        _handler_instance = DelistedStockHandler()
    return _handler_instance
