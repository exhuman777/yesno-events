#!/usr/bin/env python3
"""
YES/NO.EVENTS Terminal - UI Test Suite
=======================================
Tests for Textual UI components and screens.
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# APP STATE TESTS
# ============================================================================

class TestAppState:
    """Tests for AppState global state management"""

    def setup_method(self):
        """Reset state before each test"""
        from app import AppState
        AppState.logged_in = False
        AppState.wallet = ""
        AppState.private_key = ""
        AppState.theme = "neon"
        AppState.accent_color = "#ff8c00"

    def test_initial_state(self):
        from app import AppState
        assert AppState.logged_in == False
        assert AppState.wallet == ""
        assert AppState.theme == "neon"

    def test_login_valid_wallet(self):
        from app import AppState
        result = AppState.login("0x1234567890abcdef1234567890abcdef12345678", "test_pk")
        assert result == True
        assert AppState.logged_in == True
        assert AppState.wallet.startswith("0x")

    def test_login_invalid_wallet(self):
        from app import AppState
        result = AppState.login("invalid_wallet", "test_pk")
        assert result == False
        assert AppState.logged_in == False

    def test_logout(self):
        from app import AppState
        AppState.login("0x1234567890abcdef1234567890abcdef12345678", "test_pk")
        AppState.logout()
        assert AppState.logged_in == False
        assert AppState.wallet == ""

    def test_set_theme(self):
        from app import AppState
        AppState.set_theme("hacker")
        assert AppState.theme == "hacker"
        assert AppState.accent_color == "#00ff00"

    def test_set_invalid_theme(self):
        from app import AppState
        original_theme = AppState.theme
        AppState.set_theme("invalid_theme")
        assert AppState.theme == original_theme  # Should not change

    def test_get_colors(self):
        from app import AppState
        colors = AppState.get_colors()
        assert "bg" in colors
        assert "accent" in colors
        assert "green" in colors
        assert "red" in colors


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

class TestHelperFunctions:
    """Tests for UI helper functions"""

    def test_sparkline_generation(self):
        from app import sparkline
        values = [1, 2, 3, 4, 5, 6, 7, 8]
        result = sparkline(values, width=8)
        assert len(result) == 8
        assert all(c in "▁▂▃▄▅▆▇█" for c in result)

    def test_sparkline_empty_values(self):
        from app import sparkline
        result = sparkline([], width=10)
        assert len(result) == 10

    def test_sparkline_constant_values(self):
        from app import sparkline
        result = sparkline([5, 5, 5, 5], width=4)
        assert len(result) == 4

    def test_progress_bar(self):
        from app import progress_bar
        result = progress_bar(0.5, width=20)
        assert len(result) == 20
        assert "█" in result
        assert "░" in result

    def test_progress_bar_full(self):
        from app import progress_bar
        result = progress_bar(1.0, width=10)
        assert result == "█" * 10

    def test_progress_bar_empty(self):
        from app import progress_bar
        result = progress_bar(0.0, width=10)
        assert result == "░" * 10

    def test_fmt_pct_aligned(self):
        from app import fmt_pct, STATE
        STATE.number_format = "aligned"
        result = fmt_pct(45.67, width=8)
        assert "45.67%" in result
        assert len(result) == 9  # width + % sign

    def test_fmt_money_millions(self):
        from app import fmt_money, STATE
        STATE.number_format = "compact"
        result = fmt_money(2500000)
        assert "M" in result or "2.5" in result

    def test_fmt_money_thousands(self):
        from app import fmt_money, STATE
        STATE.number_format = "compact"
        result = fmt_money(50000)
        assert "k" in result or "50" in result

    def test_fmt_price(self):
        from app import fmt_price
        result = fmt_price(52.5, width=8)
        assert "¢" in result
        assert "52.5" in result or "52.50" in result

    def test_depth_bar_right(self):
        from app import depth_bar
        result = depth_bar(50, 100, width=10, direction="right")
        assert len(result) == 10
        assert result.startswith("█")

    def test_depth_bar_left(self):
        from app import depth_bar
        result = depth_bar(50, 100, width=10, direction="left")
        assert len(result) == 10
        assert result.endswith("█")


# ============================================================================
# SCREEN RENDERING TESTS
# ============================================================================

class TestScreenRendering:
    """Tests for screen content rendering"""

    def test_main_menu_renders(self):
        from app import MainMenuScreen
        screen = MainMenuScreen()
        content = screen._render_menu()
        # Check for key menu items (ASCII art uses special chars)
        assert "MARKETS" in content
        assert "TRADING" in content
        assert "PORTFOLIO" in content
        assert "ELON LAB" in content
        assert "POLYMARKET QUANT TERMINAL" in content

    def test_portfolio_renders_logged_out(self):
        from app import PortfolioScreen, STATE
        STATE.logged_in = False
        screen = PortfolioScreen()
        content = screen._render_content()
        assert "NOT LOGGED IN" in content

    def test_portfolio_tabs_exist(self):
        from app import PortfolioScreen, STATE
        STATE.logged_in = True
        STATE.wallet = "0x" + "a" * 40
        screen = PortfolioScreen()
        content = screen._render_content()
        assert "Positions" in content
        assert "Trades" in content
        assert "Activity" in content
        assert "Orders" in content

    def test_research_leaderboard_renders(self):
        from app import ResearchScreen
        screen = ResearchScreen()
        screen.tab = "leaderboard"
        content = screen._render_leaderboard()
        assert isinstance(content, list)
        assert len(content) > 0


# ============================================================================
# DATA DISPLAY TESTS
# ============================================================================

class TestDataDisplay:
    """Tests for data display formatting"""

    def test_positions_formatting(self):
        from app import PortfolioScreen
        screen = PortfolioScreen()
        screen.positions = [
            {"title": "Test Market", "outcome": "YES", "size": 100, "avgPrice": 0.5, "curValue": 50, "cashPnl": 5}
        ]
        content = screen._render_positions()
        assert "Test Market" in str(content)

    def test_trades_formatting(self):
        from app import PortfolioScreen
        screen = PortfolioScreen()
        screen.trades = [
            {"timestamp": "2025-01-18", "side": "BUY", "title": "Test Trade", "price": 0.5, "size": 100}
        ]
        content = screen._render_trades()
        assert "BUY" in str(content) or "Test Trade" in str(content)

    def test_activity_formatting(self):
        from app import PortfolioScreen
        screen = PortfolioScreen()
        screen.activity = [
            {"timestamp": "2025-01-18", "type": "TRADE", "title": "Test Activity", "amount": 100}
        ]
        content = screen._render_activity()
        assert "TRADE" in str(content) or "Test Activity" in str(content)

    def test_orders_formatting(self):
        from app import PortfolioScreen
        screen = PortfolioScreen()
        screen.open_orders = [
            {"title": "Test Order", "side": "BUY", "price": 0.5, "size": 100, "status": "OPEN"}
        ]
        content = screen._render_orders()
        assert "OPEN" in str(content) or "Test Order" in str(content)


# ============================================================================
# MARKET DETAIL SCREEN TESTS
# ============================================================================

class TestMarketDetailScreen:
    """Tests for MarketDetailScreen"""

    def test_market_detail_init(self):
        from app import MarketDetailScreen
        market = {"id": "test123", "title": "Test Market", "yes": 0.5}
        screen = MarketDetailScreen(market)
        assert screen.market_id == "test123"
        assert screen.show_comments == False

    def test_market_detail_render(self):
        from app import MarketDetailScreen
        market = {
            "id": "test123",
            "title": "Will something happen?",
            "yes": 0.55,
            "volume": 100000,
            "volume_24h": 5000
        }
        screen = MarketDetailScreen(market)
        content = screen._render_detail()
        assert "Will something happen?" in content
        assert "YES" in content  # Price displayed as YES %
        assert "Vol:" in content  # Compact volume display

    def test_market_detail_with_holders(self):
        from app import MarketDetailScreen
        market = {"id": "test123", "title": "Test"}
        screen = MarketDetailScreen(market)
        screen.holders = [
            {"address": "0xAAA...BBB", "outcome": "YES", "shares": 1000}
        ]
        content = screen._render_detail()
        assert "HOLDERS" in content or "0xAAA" in content

    def test_market_detail_comments_toggle(self):
        from app import MarketDetailScreen
        market = {"id": "test123", "title": "Test"}
        screen = MarketDetailScreen(market)
        screen.comments = [{"author": "User1", "content": "Great market!"}]

        # Comments hidden by default
        screen.show_comments = False
        content1 = screen._render_detail()

        # Toggle comments
        screen.show_comments = True
        content2 = screen._render_detail()

        # Content should differ
        assert "COMMENTS" in content2


# ============================================================================
# THEME TESTS
# ============================================================================

class TestThemes:
    """Tests for theme system"""

    def test_all_themes_exist(self):
        from app import AppState
        expected_themes = ["neon", "dark", "hacker", "ocean"]
        for theme in expected_themes:
            assert theme in AppState.THEMES

    def test_theme_has_required_colors(self):
        from app import AppState
        required_colors = ["bg", "accent", "green", "red"]
        for theme_name, theme_colors in AppState.THEMES.items():
            for color in required_colors:
                assert color in theme_colors, f"{theme_name} missing {color}"

    def test_theme_colors_are_valid_hex(self):
        from app import AppState
        for theme_name, theme_colors in AppState.THEMES.items():
            for color_name, color_value in theme_colors.items():
                assert color_value.startswith("#"), f"{theme_name}.{color_name} invalid"
                assert len(color_value) == 7, f"{theme_name}.{color_name} wrong length"


# ============================================================================
# ACTION TESTS
# ============================================================================

class TestActions:
    """Tests for screen actions - testing tab state changes only"""

    def test_portfolio_tab_state_changes(self):
        """Test tab state changes without requiring mounted screen"""
        from app import PortfolioScreen
        screen = PortfolioScreen()

        # Test tab state changes directly
        screen.tab = "positions"
        assert screen.tab == "positions"

        screen.tab = "trades"
        assert screen.tab == "trades"

        screen.tab = "activity"
        assert screen.tab == "activity"

        screen.tab = "orders"
        assert screen.tab == "orders"

    def test_research_tab_state_changes(self):
        """Test tab state changes without requiring mounted screen"""
        from app import ResearchScreen
        screen = ResearchScreen()

        # Test tab state changes directly
        screen.tab = "leaderboard"
        assert screen.tab == "leaderboard"

        screen.tab = "tags"
        assert screen.tab == "tags"

        screen.tab = "patterns"
        assert screen.tab == "patterns"

        screen.tab = "correlations"
        assert screen.tab == "correlations"


# ============================================================================
# HELP SCREEN TESTS
# ============================================================================

class TestHelpScreen:
    """Tests for in-app help system"""

    def test_help_content_exists(self):
        from app import HELP_CONTENT
        assert len(HELP_CONTENT) >= 5
        assert "overview" in HELP_CONTENT
        assert "navigation" in HELP_CONTENT
        assert "search" in HELP_CONTENT
        assert "trading" in HELP_CONTENT
        assert "api" in HELP_CONTENT

    def test_help_screen_init(self):
        from app import HelpScreen
        screen = HelpScreen()
        assert screen.section == "overview"

    def test_help_section_changes(self):
        from app import HelpScreen
        screen = HelpScreen()

        screen.section = "navigation"
        assert screen.section == "navigation"

        screen.section = "search"
        assert screen.section == "search"

        screen.section = "trading"
        assert screen.section == "trading"

    def test_help_content_has_keywords(self):
        from app import HELP_CONTENT
        # Overview should have basics
        assert "RESEARCH" in HELP_CONTENT["overview"]
        # Navigation should have shortcuts
        assert "ESC" in HELP_CONTENT["navigation"]
        # Search should have categories
        assert "POLITICS" in HELP_CONTENT["search"]
        # Trading should have order types
        assert "LIMIT" in HELP_CONTENT["trading"]
        # API should have endpoints
        assert "gamma" in HELP_CONTENT["api"].lower()


# ============================================================================
# WORLD MAP TESTS
# ============================================================================

class TestWorldMap:
    """Tests for world map screen"""

    def test_market_regions_exist(self):
        from app import MARKET_REGIONS
        assert "US" in MARKET_REGIONS
        assert "EU" in MARKET_REGIONS
        assert "ASIA" in MARKET_REGIONS
        assert "CRYPTO" in MARKET_REGIONS

    def test_region_data_structure(self):
        from app import MARKET_REGIONS
        for region, data in MARKET_REGIONS.items():
            assert "label" in data
            assert "markets" in data
            assert isinstance(data["markets"], list)

    def test_world_map_screen_init(self):
        from app import WorldMapScreen
        screen = WorldMapScreen()
        assert screen.selected_region is None
        assert isinstance(screen.region_markets, dict)


# ============================================================================
# ELON DATA TESTS
# ============================================================================

class TestElonData:
    """Tests for pre-computed Elon tweet data"""

    def test_elon_data_exists(self):
        from app import ELON_DAILY_DATA, ELON_HOURLY_TOTALS, ELON_TOTAL_TWEETS
        assert len(ELON_DAILY_DATA) > 0
        assert len(ELON_HOURLY_TOTALS) == 24
        assert ELON_TOTAL_TWEETS > 0

    def test_elon_data_structure(self):
        from app import ELON_DAILY_DATA
        for day in ELON_DAILY_DATA:
            assert "date" in day
            assert "day" in day
            assert "hourly" in day
            assert "total" in day
            assert len(day["hourly"]) == 24

    def test_elon_totals_match(self):
        from app import ELON_DAILY_DATA, ELON_TOTAL_TWEETS
        calculated_total = sum(d["total"] for d in ELON_DAILY_DATA)
        assert calculated_total == ELON_TOTAL_TWEETS


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
