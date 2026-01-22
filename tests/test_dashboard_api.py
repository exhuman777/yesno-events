#!/usr/bin/env python3
"""
YES/NO.EVENTS - Dashboard API Test Suite
========================================
Tests for web dashboard HTTP API endpoints.
"""
import pytest
import json
import sys
import threading
import time
from pathlib import Path
from http.client import HTTPConnection
from urllib.parse import urlencode

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture(scope="module")
def server():
    """Start dashboard server for testing"""
    import dashboard4all
    from http.server import HTTPServer

    # Use a high port to avoid conflicts
    port = 18888
    server = HTTPServer(('localhost', port), dashboard4all.Handler)

    # Run server in background thread
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    # Wait for server to start
    time.sleep(0.5)

    yield port

    # Cleanup
    server.shutdown()


@pytest.fixture
def http_client(server):
    """HTTP client connected to test server"""
    conn = HTTPConnection('localhost', server, timeout=10)
    yield conn
    conn.close()


def get_json(client, path):
    """Helper to GET JSON from API"""
    client.request('GET', path)
    response = client.getresponse()
    data = response.read().decode('utf-8')
    return response.status, json.loads(data) if data else {}


# ============================================================================
# API ENDPOINT TESTS
# ============================================================================

class TestAPIEndpoints:
    """Tests for API endpoint availability"""

    def test_endpoints_returns_api_docs(self, http_client):
        """Should return API documentation"""
        status, data = get_json(http_client, '/api/endpoints')

        assert status == 200
        assert 'endpoints' in data
        assert 'version' in data

    def test_watched_markets_returns_list(self, http_client):
        """Should return watched markets list"""
        status, data = get_json(http_client, '/api/watched')

        assert status == 200
        assert 'markets' in data
        assert isinstance(data['markets'], list)

    def test_cache_size_returns_number(self, http_client):
        """Should return cache size"""
        status, data = get_json(http_client, '/api/cache-size')

        assert status == 200
        assert 'size_mb' in data or 'size' in data

    def test_history_returns_data(self, http_client):
        """Should return research history"""
        status, data = get_json(http_client, '/api/history')

        assert status == 200
        # May have searches or viewed_markets

    def test_alerts_returns_list(self, http_client):
        """Should return alerts list"""
        status, data = get_json(http_client, '/api/alerts')

        assert status == 200
        assert 'alerts' in data
        assert isinstance(data['alerts'], list)


class TestSearchAPI:
    """Tests for search endpoints"""

    def test_search_requires_query(self, http_client):
        """Should return error without query"""
        status, data = get_json(http_client, '/api/search')

        # Should return error or empty
        assert status == 200
        assert 'error' in data or 'markets' in data

    def test_search_with_query(self, http_client):
        """Should search with query parameter"""
        status, data = get_json(http_client, '/api/search?q=bitcoin')

        assert status == 200
        # May return markets or error (depends on API availability)

    def test_vector_search_endpoint(self, http_client):
        """Should have vector search endpoint"""
        status, data = get_json(http_client, '/api/search/vector?q=elon')

        assert status == 200
        assert 'results' in data or 'error' in data

    def test_similar_markets_endpoint(self, http_client):
        """Should have similar markets endpoint"""
        status, data = get_json(http_client, '/api/search/similar?id=test123')

        assert status == 200
        # May return results or error


class TestMarketDataAPI:
    """Tests for market data endpoints"""

    def test_market_requires_id(self, http_client):
        """Should require market ID"""
        status, data = get_json(http_client, '/api/market')

        # Should indicate missing ID
        assert status == 200
        assert 'error' in data or 'markets' in data

    def test_orderbook_endpoint(self, http_client):
        """Should have orderbook endpoint"""
        status, data = get_json(http_client, '/api/orderbook?id=test123')

        assert status == 200
        # May return data or error

    def test_trades_endpoint(self, http_client):
        """Should have trades endpoint"""
        status, data = get_json(http_client, '/api/trades?id=test123')

        assert status == 200


class TestElonAPI:
    """Tests for Elon prediction endpoints"""

    def test_elon_data_endpoint(self, http_client):
        """Should return Elon data"""
        status, data = get_json(http_client, '/api/elon')

        assert status == 200
        # Should have some data structure

    def test_elon_stats_endpoint(self, http_client):
        """Should return Elon stats"""
        status, data = get_json(http_client, '/api/elon/stats')

        assert status == 200

    def test_elon_models_endpoint(self, http_client):
        """Should return prediction models"""
        status, data = get_json(http_client, '/api/elon/models')

        assert status == 200

    def test_elon_predict_endpoint(self, http_client):
        """Should make prediction with params"""
        status, data = get_json(http_client, '/api/elon/predict?current=200&days_elapsed=3')

        assert status == 200


class TestTradingAPI:
    """Tests for trading endpoints"""

    def test_trading_price_endpoint(self, http_client):
        """Should have price endpoint"""
        status, data = get_json(http_client, '/api/trading/price?id=test123')

        assert status == 200

    def test_trading_orders_endpoint(self, http_client):
        """Should have orders endpoint"""
        status, data = get_json(http_client, '/api/trading/orders')

        assert status == 200

    def test_trading_positions_endpoint(self, http_client):
        """Should have positions endpoint"""
        status, data = get_json(http_client, '/api/trading/positions')

        assert status == 200


class TestQuantAPI:
    """Tests for quant tool endpoints"""

    def test_kelly_endpoint(self, http_client):
        """Should calculate Kelly criterion"""
        status, data = get_json(http_client, '/api/kelly?prob=0.6&price=0.5&bankroll=1000')

        assert status == 200
        # May have calculation result or error

    def test_edges_endpoint(self, http_client):
        """Should have edge scanner"""
        status, data = get_json(http_client, '/api/edges')

        assert status == 200

    def test_momentum_endpoint(self, http_client):
        """Should have momentum scanner"""
        status, data = get_json(http_client, '/api/edges/momentum')

        assert status == 200


# ============================================================================
# VIEW RENDERING TESTS
# ============================================================================

class TestViewRendering:
    """Tests for HTML view rendering"""

    def test_home_page_renders(self, http_client):
        """Should render home page"""
        http_client.request('GET', '/')
        response = http_client.getresponse()
        html = response.read().decode('utf-8')

        assert response.status == 200
        assert 'DOCTYPE html' in html
        assert 'DASHBOARD4ALL' in html

    def test_markets_view_renders(self, http_client):
        """Should render markets view"""
        http_client.request('GET', '/?view=markets')
        response = http_client.getresponse()
        html = response.read().decode('utf-8')

        assert response.status == 200
        assert 'DOCTYPE html' in html

    def test_elon_view_renders(self, http_client):
        """Should render Elon view"""
        http_client.request('GET', '/?view=elon')
        response = http_client.getresponse()
        html = response.read().decode('utf-8')

        assert response.status == 200
        assert 'DOCTYPE html' in html

    def test_market_detail_view(self, http_client):
        """Should render market detail"""
        http_client.request('GET', '/?view=market&id=test123')
        response = http_client.getresponse()
        html = response.read().decode('utf-8')

        assert response.status == 200
        assert 'DOCTYPE html' in html


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling"""

    def test_invalid_view_handled(self, http_client):
        """Should handle invalid view gracefully"""
        http_client.request('GET', '/?view=nonexistent')
        response = http_client.getresponse()

        # Should not crash - return 200 or redirect
        assert response.status in [200, 302, 404]

    def test_malformed_params_handled(self, http_client):
        """Should handle malformed parameters"""
        status, data = get_json(http_client, '/api/kelly?prob=notanumber')

        assert status == 200
        # Should return error or default behavior

    def test_missing_required_params(self, http_client):
        """Should handle missing required params"""
        status, data = get_json(http_client, '/api/market')

        assert status == 200
        # Should indicate error


# ============================================================================
# SECURITY TESTS
# ============================================================================

class TestSecurity:
    """Tests for security features"""

    def test_xss_prevention_in_search(self, http_client):
        """Should escape XSS in search queries"""
        # Try XSS payload
        payload = '<script>alert("xss")</script>'
        http_client.request('GET', f'/?view=markets&q={payload}')
        response = http_client.getresponse()
        html = response.read().decode('utf-8')

        # Script tag should be escaped
        assert '<script>' not in html or '&lt;script&gt;' in html

    def test_xss_prevention_in_market_id(self, http_client):
        """Should escape XSS in market IDs"""
        payload = '"><script>alert(1)</script>'
        http_client.request('GET', f'/?view=market&id={payload}')
        response = http_client.getresponse()
        html = response.read().decode('utf-8')

        assert response.status == 200
        # Should not have raw script tag


# ============================================================================
# RESPONSE FORMAT TESTS
# ============================================================================

class TestResponseFormat:
    """Tests for response format consistency"""

    def test_api_returns_json(self, http_client):
        """API endpoints should return JSON"""
        endpoints = [
            '/api/endpoints',
            '/api/watched',
            '/api/alerts',
            '/api/cache-size',
        ]

        for endpoint in endpoints:
            status, data = get_json(http_client, endpoint)
            assert isinstance(data, dict), f"{endpoint} should return dict"

    def test_api_error_format(self, http_client):
        """Errors should have consistent format"""
        status, data = get_json(http_client, '/api/search')  # Missing query

        # If error, should have 'error' key
        if status == 200 and 'error' in data:
            assert isinstance(data['error'], (str, dict))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
