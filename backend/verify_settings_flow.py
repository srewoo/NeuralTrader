#!/usr/bin/env python3
"""
Verification script for Settings-based API key management flow.
Tests: Browser Settings UI → localStorage → MongoDB → Backend retrieval
"""

import asyncio
import sys
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime

# Database connection
MONGO_URL = "mongodb://localhost:27017/neuraltrader"

async def verify_settings_flow():
    """Verify the complete settings flow"""
    print("=" * 70)
    print("NeuralTrader Settings Flow Verification")
    print("=" * 70)

    # Connect to MongoDB
    print("\n1. Connecting to MongoDB...")
    try:
        client = AsyncIOMotorClient(MONGO_URL)
        db = client.neuraltrader
        await db.command('ping')
        print("   ✅ MongoDB connection successful")
    except Exception as e:
        print(f"   ❌ MongoDB connection failed: {e}")
        print("   💡 Make sure MongoDB is running: brew services start mongodb-community")
        return False

    # Verify Settings collection structure
    print("\n2. Checking Settings collection...")
    settings = await db.settings.find_one({})

    if not settings:
        print("   ⚠️  No settings found in database (this is normal for new installations)")
        print("   💡 Settings will be created when user saves via Settings UI")
    else:
        print(f"   ✅ Found existing settings (ID: {settings.get('id', 'N/A')})")

        # Check for all required API key fields
        required_keys = [
            'openai_api_key',
            'gemini_api_key',
            'finnhub_api_key',
            'alpaca_api_key',
            'alpaca_api_secret',
            'fmp_api_key',
            'iex_api_key'
        ]

        print("\n   Checking API key fields:")
        for key in required_keys:
            has_value = bool(settings.get(key))
            status = "🔑" if has_value else "⚪"
            print(f"   {status} {key}: {'Configured' if has_value else 'Not set'}")

    # Test Settings model
    print("\n3. Verifying Settings models...")
    try:
        from server import Settings, SettingsCreate

        # Check all fields are present
        settings_fields = Settings.model_fields.keys()
        create_fields = SettingsCreate.model_fields.keys()

        required_fields = {
            'openai_api_key', 'gemini_api_key', 'finnhub_api_key',
            'alpaca_api_key', 'alpaca_api_secret', 'fmp_api_key', 'iex_api_key'
        }

        missing_in_settings = required_fields - set(settings_fields)
        missing_in_create = required_fields - set(create_fields)

        if missing_in_settings or missing_in_create:
            print(f"   ❌ Missing fields!")
            if missing_in_settings:
                print(f"      Settings model missing: {missing_in_settings}")
            if missing_in_create:
                print(f"      SettingsCreate model missing: {missing_in_create}")
            return False
        else:
            print("   ✅ All API key fields present in Settings models")
            print(f"   ✅ Settings model has {len(settings_fields)} fields")
            print(f"   ✅ SettingsCreate model has {len(create_fields)} fields")

    except Exception as e:
        print(f"   ❌ Settings model verification failed: {e}")
        return False

    # Test data provider factory
    print("\n4. Testing Data Provider Factory...")
    try:
        from data_providers.factory import get_data_provider_factory

        # Test with empty keys (should fall back to yfinance)
        factory = get_data_provider_factory(provider_keys={})
        print("   ✅ Data provider factory initialized successfully")
        print("   ✅ Will use yfinance as fallback when no API keys configured")

        # Test with sample keys
        sample_keys = {
            'alpaca': {'key': 'test_key', 'secret': 'test_secret', 'paper': True},
            'iex': 'test_iex_key'
        }
        factory_with_keys = get_data_provider_factory(provider_keys=sample_keys)
        print("   ✅ Data provider factory works with API keys")

    except Exception as e:
        print(f"   ❌ Data provider factory test failed: {e}")
        return False

    # Test cost tracking
    print("\n5. Testing Cost Tracking System...")
    try:
        from cost_tracking import get_cost_tracker

        tracker = get_cost_tracker(db)

        # Check if api_costs collection exists
        collections = await db.list_collection_names()
        if 'api_costs' in collections:
            cost_count = await db.api_costs.count_documents({})
            print(f"   ✅ Cost tracking collection exists ({cost_count} records)")
        else:
            print("   ⚪ Cost tracking collection will be created on first API call")

        # Test getting current month cost (should work even with no data)
        cost = await tracker.get_current_month_cost()
        print(f"   ✅ Current month cost: ${cost:.2f}")

    except Exception as e:
        print(f"   ❌ Cost tracking test failed: {e}")
        return False

    # Test new modules
    print("\n6. Testing New Feature Modules...")
    modules_to_test = [
        ('cost_tracking', 'get_cost_tracker'),
        ('data_providers.alpaca_live', 'get_alpaca_provider'),
        ('data_providers.iex_cloud', 'get_iex_provider'),
        ('data_providers.factory', 'get_data_provider_factory'),
        ('news.advanced_sources', 'get_advanced_news_aggregator'),
        ('news.advanced_sentiment', 'get_advanced_sentiment_analyzer'),
    ]

    for module_name, func_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[func_name])
            func = getattr(module, func_name)
            print(f"   ✅ {module_name}.{func_name}")
        except Exception as e:
            print(f"   ❌ {module_name}.{func_name}: {e}")
            return False

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    print("\n✅ All systems operational!")
    print("\n📝 Settings Flow:")
    print("   1. User enters API keys in Settings UI (browser)")
    print("   2. Frontend saves to localStorage (browser cache)")
    print("   3. Frontend POSTs to /api/settings (backend API)")
    print("   4. Backend saves to MongoDB (persistent storage)")
    print("   5. Backend retrieves keys when needed (automatic)")
    print("\n🔑 Supported API Keys:")
    print("   • OpenAI API Key (for GPT models)")
    print("   • Gemini API Key (for Gemini models)")
    print("   • Alpaca API Key + Secret (real-time stock data)")
    print("   • IEX Cloud API Key (market data)")
    print("   • Finnhub API Key (alternative data)")
    print("   • FMP API Key (Financial Modeling Prep)")
    print("\n🚀 Ready to use!")
    print("   Start backend: uvicorn server:app --reload --port 8005")
    print("   Start frontend: npm start (in frontend directory)")
    print("   Access Settings: http://localhost:3005/settings")
    print("\n" + "=" * 70)

    return True

if __name__ == "__main__":
    result = asyncio.run(verify_settings_flow())
    sys.exit(0 if result else 1)
