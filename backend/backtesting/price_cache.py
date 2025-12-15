"""
Price Cache using SQLite
Caches historical price data for fast backtesting
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging
import yfinance as yf
from pathlib import Path

logger = logging.getLogger(__name__)


def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"  # Default to NSE
    return symbol


class PriceCache:
    """
    SQLite-based cache for historical price data
    """
    
    def __init__(self, db_path: str = "price_cache.db"):
        """
        Initialize price cache
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database and create tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create prices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adjusted_close REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_symbol_date 
                ON prices(symbol, date)
            """)
            
            # Create metadata table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_updated TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    record_count INTEGER
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info(f"Price cache database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def get_prices(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        Get historical prices from cache or fetch from yfinance

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_refresh: Force refresh from yfinance

        Returns:
            DataFrame with OHLCV data (with DatetimeIndex)
        """
        try:
            # Check if data exists in cache
            if not force_refresh and self._has_data(symbol, start_date, end_date):
                logger.info(f"Loading {symbol} from cache")
                return self._load_from_cache(symbol, start_date, end_date)

            # Fetch from yfinance (REAL API CALL)
            logger.info(f"Fetching {symbol} from yfinance")
            df = self._fetch_from_yfinance(symbol, start_date, end_date)

            # Save to cache using the cached version with date column
            if not df.empty and hasattr(self, '_last_fetch_for_cache'):
                self._save_to_cache(symbol, self._last_fetch_for_cache)
                del self._last_fetch_for_cache

            return df

        except Exception as e:
            logger.error(f"Failed to get prices for {symbol}: {e}")
            return pd.DataFrame()
    
    def _has_data(self, symbol: str, start_date: str, end_date: str) -> bool:
        """Check if data exists in cache for the date range"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check metadata
            cursor.execute("""
                SELECT start_date, end_date, last_updated 
                FROM cache_metadata 
                WHERE symbol = ?
            """, (symbol.upper(),))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return False
            
            cached_start, cached_end, last_updated = result
            
            # Check if cached range covers requested range
            if cached_start <= start_date and cached_end >= end_date:
                # Check if cache is not too old (7 days)
                last_update = datetime.fromisoformat(last_updated)
                if datetime.now() - last_update < timedelta(days=7):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    def _load_from_cache(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Load data from cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT date, open, high, low, close, volume, adjusted_close
                FROM prices
                WHERE symbol = ? AND date >= ? AND date <= ?
                ORDER BY date
            """
            
            df = pd.read_sql_query(
                query,
                conn,
                params=(symbol.upper(), start_date, end_date)
            )
            
            conn.close()
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load from cache: {e}")
            return pd.DataFrame()
    
    def _fetch_from_yfinance(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data from yfinance (REAL API CALL)"""
        try:
            # Add .NS suffix for Indian stocks
            ticker_symbol = get_indian_stock_suffix(symbol)
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return pd.DataFrame()

            # Keep a copy with date column for caching
            df_for_cache = df.reset_index()
            df_for_cache.rename(columns={'Date': 'date'}, inplace=True)

            # Add adjusted close if not present
            if 'Adj Close' not in df_for_cache.columns:
                df_for_cache['Adj Close'] = df_for_cache['Close']

            # Store the cache version for later saving
            self._last_fetch_for_cache = df_for_cache

            # Return DataFrame with datetime index (for consistency with cache)
            # yfinance already returns with DatetimeIndex
            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']

            return df

        except Exception as e:
            logger.error(f"Failed to fetch from yfinance: {e}")
            return pd.DataFrame()
    
    def _save_to_cache(self, symbol: str, df: pd.DataFrame):
        """Save data to cache"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            symbol_upper = symbol.upper()
            
            # Prepare data for insertion
            records = []
            for idx, row in df.iterrows():
                date_str = row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date'])
                records.append((
                    symbol_upper,
                    date_str,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume']),
                    float(row.get('Adj Close', row['Close']))
                ))
            
            # Insert or replace records
            cursor.executemany("""
                INSERT OR REPLACE INTO prices 
                (symbol, date, open, high, low, close, volume, adjusted_close)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, records)
            
            # Update metadata
            start_date = df['date'].min().strftime('%Y-%m-%d') if isinstance(df['date'].min(), pd.Timestamp) else str(df['date'].min())
            end_date = df['date'].max().strftime('%Y-%m-%d') if isinstance(df['date'].max(), pd.Timestamp) else str(df['date'].max())
            
            cursor.execute("""
                INSERT OR REPLACE INTO cache_metadata 
                (symbol, last_updated, start_date, end_date, record_count)
                VALUES (?, ?, ?, ?, ?)
            """, (
                symbol_upper,
                datetime.now().isoformat(),
                start_date,
                end_date,
                len(records)
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cached {len(records)} records for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to save to cache: {e}")
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear cache for a symbol or all symbols
        
        Args:
            symbol: Symbol to clear (None = clear all)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute("DELETE FROM prices WHERE symbol = ?", (symbol.upper(),))
                cursor.execute("DELETE FROM cache_metadata WHERE symbol = ?", (symbol.upper(),))
                logger.info(f"Cleared cache for {symbol}")
            else:
                cursor.execute("DELETE FROM prices")
                cursor.execute("DELETE FROM cache_metadata")
                logger.info("Cleared all cache")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get symbol count
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices")
            symbol_count = cursor.fetchone()[0]
            
            # Get total records
            cursor.execute("SELECT COUNT(*) FROM prices")
            total_records = cursor.fetchone()[0]
            
            # Get database size
            db_size = Path(self.db_path).stat().st_size / (1024 * 1024)  # MB
            
            conn.close()
            
            return {
                "symbols_cached": symbol_count,
                "total_records": total_records,
                "database_size_mb": round(db_size, 2)
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}


# Global instance
_price_cache_instance = None


def get_price_cache() -> PriceCache:
    """Get or create global PriceCache instance"""
    global _price_cache_instance
    if _price_cache_instance is None:
        _price_cache_instance = PriceCache()
    return _price_cache_instance

