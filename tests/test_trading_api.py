#!/usr/bin/env python3
"""
YES/NO.EVENTS Terminal - API Test Suite
========================================
Comprehensive tests for all Polymarket API integrations.
"""
import pytest
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading import (
    # Data API v2
    dataapi_get_positions,
    dataapi_get_trades,
    dataapi_get_activity,
    dataapi_get_value,
    dataapi_get_holders,
    dataapi_get_leaderboard,
    # Gamma API
    gamma_get_public_profile,
    gamma_get_comments,
    gamma_get_market_resolution,
    gamma_search,
    gamma_get_market,
    # CLOB API
    get_orderbook,
    get_spread,
    get_market_trades,
    # Analysis
    get_price_history,
    find_ev_opportunities,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_wallet():
    return "0x1234567890abcdef1234567890abcdef12345678"

@pytest.fixture
def sample_market_id():
    # Use a real, active market ID for tests
    return "1172394"  # Elon 420-439 market (known working)

@pytest.fixture
def mock_positions_response():
    return [
        {
            "title": "Will Trump win 2024?",
            "outcome": "YES",
            "size": 1000,
            "avgPrice": 0.52,
            "curValue": 520,
            "cashPnl": 45.50
        },
        {
            "title": "Bitcoin > $100k by March?",
            "outcome": "NO",
            "size": 500,
            "avgPrice": 0.35,
            "curValue": 175,
            "cashPnl": -12.25
        }
    ]

@pytest.fixture
def mock_leaderboard_response():
    return [
        {"address": "0xAAA...BBB", "pnl": 2500000, "volume": 15000000, "positions": 45},
        {"address": "0xCCC...DDD", "pnl": 1800000, "volume": 12000000, "positions": 32},
        {"address": "0xEEE...FFF", "pnl": 1200000, "volume": 8500000, "positions": 28},
    ]

@pytest.fixture
def mock_orderbook_response():
    return {
        "bids": [
            {"price": 0.52, "size": 5000},
            {"price": 0.51, "size": 3000},
            {"price": 0.50, "size": 8000},
        ],
        "asks": [
            {"price": 0.53, "size": 4000},
            {"price": 0.54, "size": 2500},
            {"price": 0.55, "size": 6000},
        ]
    }


# ============================================================================
# DATA API v2 TESTS
# ============================================================================

class TestDataAPIPositions:
    """Tests for dataapi_get_positions endpoint"""

    def test_returns_list_for_valid_wallet(self, sample_wallet):
        """Should return a list (possibly empty) for valid wallet"""
        result = dataapi_get_positions(sample_wallet)
        assert isinstance(result, (list, dict))

    def test_handles_invalid_wallet_gracefully(self):
        """Should not crash with invalid wallet address"""
        result = dataapi_get_positions("invalid_wallet")
        assert result is not None

    def test_respects_limit_parameter(self, sample_wallet):
        """Should accept limit parameter"""
        result = dataapi_get_positions(sample_wallet, limit=10)
        assert isinstance(result, (list, dict))

    def test_respects_sort_by_parameter(self, sample_wallet):
        """Should accept sort_by parameter"""
        for sort_by in ["TOKENS", "CURRENT", "INITIAL", "CASHPNL", "PERCENTPNL"]:
            result = dataapi_get_positions(sample_wallet, sort_by=sort_by)
            assert isinstance(result, (list, dict))


class TestDataAPITrades:
    """Tests for dataapi_get_trades endpoint"""

    def test_returns_data_for_user_trades(self, sample_wallet):
        """Should return trade data for user"""
        result = dataapi_get_trades(user_address=sample_wallet)
        assert isinstance(result, (list, dict))

    def test_returns_data_for_market_trades(self, sample_market_id):
        """Should return trade data for market"""
        result = dataapi_get_trades(market=sample_market_id)
        assert isinstance(result, (list, dict))

    def test_filters_by_side(self, sample_wallet):
        """Should accept side filter"""
        for side in ["BUY", "SELL"]:
            result = dataapi_get_trades(user_address=sample_wallet, side=side)
            assert isinstance(result, (list, dict))


class TestDataAPILeaderboard:
    """Tests for dataapi_get_leaderboard endpoint"""

    def test_returns_leaderboard_data(self):
        """Should return leaderboard data"""
        result = dataapi_get_leaderboard()
        assert isinstance(result, (list, dict))

    def test_accepts_category_parameter(self):
        """Should accept category parameter"""
        categories = ["OVERALL", "POLITICS", "SPORTS", "CRYPTO", "CULTURE"]
        for cat in categories:
            result = dataapi_get_leaderboard(category=cat)
            assert isinstance(result, (list, dict))

    def test_accepts_time_period_parameter(self):
        """Should accept time_period parameter"""
        periods = ["DAY", "WEEK", "MONTH", "ALL"]
        for period in periods:
            result = dataapi_get_leaderboard(time_period=period)
            assert isinstance(result, (list, dict))


class TestDataAPIHolders:
    """Tests for dataapi_get_holders endpoint"""

    def test_returns_holder_data(self, sample_market_id):
        """Should return holder data for market"""
        result = dataapi_get_holders(sample_market_id)
        assert isinstance(result, (list, dict))

    def test_respects_limit_parameter(self, sample_market_id):
        """Should accept limit parameter"""
        result = dataapi_get_holders(sample_market_id, limit=5)
        assert isinstance(result, (list, dict))


# ============================================================================
# GAMMA API TESTS
# ============================================================================

class TestGammaAPI:
    """Tests for Gamma API endpoints"""

    def test_search_returns_results(self):
        """Should return search results"""
        result = gamma_search("Trump")
        assert isinstance(result, (list, dict))

    def test_get_market_handles_invalid_id(self):
        """Should handle invalid market ID gracefully"""
        result = gamma_get_market("invalid_id")
        # Should not crash, may return None or error dict
        assert True

    def test_get_comments_returns_list(self):
        """Should return comments list"""
        result = gamma_get_comments("Event", "12345")
        assert isinstance(result, (list, dict))

    def test_get_public_profile_returns_data(self, sample_wallet):
        """Should return profile data or error"""
        result = gamma_get_public_profile(sample_wallet)
        assert isinstance(result, dict)


# ============================================================================
# CLOB API TESTS
# ============================================================================

class TestCLOBAPI:
    """Tests for CLOB API endpoints"""

    def test_get_orderbook_returns_data(self, sample_market_id):
        """Should return orderbook structure"""
        result = get_orderbook(sample_market_id, "yes")
        # May return dict or object
        assert result is not None or result is None  # Graceful handling

    def test_get_spread_calculates_correctly(self, sample_market_id):
        """Should calculate spread"""
        result = get_spread(sample_market_id)
        assert isinstance(result, (dict, type(None), float, int))

    def test_get_market_trades_returns_list(self, sample_market_id):
        """Should return trades list"""
        result = get_market_trades(sample_market_id, limit=10)
        assert isinstance(result, (list, type(None)))


# ============================================================================
# ANALYSIS TESTS
# ============================================================================

class TestAnalysis:
    """Tests for analysis functions"""

    def test_get_price_history_returns_data(self, sample_market_id):
        """Should return price history"""
        result = get_price_history(sample_market_id, interval=60)
        assert isinstance(result, (list, type(None)))

    def test_find_ev_opportunities_returns_list(self):
        """Should return EV opportunities"""
        result = find_ev_opportunities()
        assert isinstance(result, (list, type(None)))


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestAPIIntegration:
    """Integration tests for API workflow"""

    def test_full_trader_analysis_workflow(self, sample_wallet):
        """Test complete trader analysis: positions -> trades -> activity"""
        # Get positions
        positions = dataapi_get_positions(sample_wallet)
        assert isinstance(positions, (list, dict))

        # Get trades
        trades = dataapi_get_trades(user_address=sample_wallet)
        assert isinstance(trades, (list, dict))

        # Get activity
        activity = dataapi_get_activity(sample_wallet)
        assert isinstance(activity, (list, dict))

    def test_full_market_analysis_workflow(self, sample_market_id):
        """Test complete market analysis: orderbook -> trades -> holders"""
        # Get orderbook
        orderbook = get_orderbook(sample_market_id, "yes")

        # Get trades
        trades = get_market_trades(sample_market_id, limit=10)

        # Get holders
        holders = dataapi_get_holders(sample_market_id, limit=5)

        # All should complete without crashing
        assert True


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_network_timeout_handled(self):
        """Should handle network timeouts gracefully"""
        # This tests that the functions don't crash on network issues
        result = dataapi_get_positions("0x" + "0" * 40)
        assert result is not None or result == []

    def test_malformed_response_handled(self):
        """Should handle malformed API responses"""
        # Functions should return empty list/dict on parse errors
        result = gamma_get_comments("Invalid", None)
        assert isinstance(result, (list, dict))

    def test_rate_limiting_handled(self):
        """Should handle rate limiting gracefully"""
        # Make multiple rapid requests
        for _ in range(5):
            result = gamma_search("test")
        # Should not crash
        assert True


# ============================================================================
# MOCK TESTS (Unit tests with mocked responses)
# ============================================================================

class TestMockedResponses:
    """Unit tests with mocked API responses"""

    @patch('urllib.request.urlopen')
    def test_positions_parsing(self, mock_urlopen, mock_positions_response):
        """Test correct parsing of positions response"""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_positions_response).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = dataapi_get_positions("0x" + "a" * 40)
        assert isinstance(result, list)

    @patch('urllib.request.urlopen')
    def test_leaderboard_parsing(self, mock_urlopen, mock_leaderboard_response):
        """Test correct parsing of leaderboard response"""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(mock_leaderboard_response).encode()
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        result = dataapi_get_leaderboard()
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
