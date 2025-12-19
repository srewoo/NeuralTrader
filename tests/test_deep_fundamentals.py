
import sys
import os
import asyncio
import logging

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

# Configure logging
logging.basicConfig(level=logging.INFO)

from analysis.enhanced_analyzer import EnhancedAnalyzer

async def test_deep_fundamentals():
    print("Testing Deep Fundamentals (Forensic, Valuation, Macro)...")
    
    analyzer = EnhancedAnalyzer()
    
    # Test on a large cap stock (likely to have good financial data)
    symbol = "TCS" 
    
    print(f"\nAnalyzing {symbol}...")
    result = await analyzer.analyze_stock(symbol, include_fundamentals=True)
    
    # Check Forensic Report
    print("\n--- Forensic Analysis ---")
    forensic = result.get('deep_fundamentals', {}).get('forensic')
    if forensic and 'error' not in forensic:
        print(f"Beneish M-Score: {forensic.get('beneish_m_score')}")
        print(f"Altman Z-Score: {forensic.get('altman_z_score')}")
    else:
        print(f"Forensic Analysis Failed or Missing: {forensic}")

    # Check Valuation Report
    print("\n--- Valuation Analysis ---")
    valuation = result.get('deep_fundamentals', {}).get('valuation')
    if valuation and 'error' not in valuation:
        print(f"Fair Value (DCF): {valuation.get('fair_value_dcf')}")
        print(f"Fair Value (Graham): {valuation.get('fair_value_graham')}")
        print(f"Signals: {valuation.get('signals')}")
    else:
        print(f"Valuation Analysis Failed or Missing: {valuation}")

    # Check Macro
    print("\n--- Macro Analysis ---")
    macro = result.get('deep_fundamentals', {}).get('macro')
    print(macro)
    
    # Check Final Recommendation
    print("\n--- Final Recommendation ---")
    summary = result.get('analysis_summary')
    print(f"Recommendation: {summary.get('recommendation')}")
    print(f"Confidence: {summary.get('confidence')}")

if __name__ == "__main__":
    asyncio.run(test_deep_fundamentals())
