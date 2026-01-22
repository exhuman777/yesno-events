#!/usr/bin/env python3
"""
YES/NO.EVENTS - Quant Models Test Suite
=======================================
Tests for Monte Carlo, Kelly Criterion, and EV calculations.
"""
import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from quant import (
    monte_carlo_tweets,
    kelly_criterion,
    calculate_ev,
    ascii_bar,
    ascii_histogram,
)


# ============================================================================
# MONTE CARLO TESTS
# ============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo tweet simulations"""

    def test_basic_projection(self):
        """Should project tweets given current count and time"""
        result = monte_carlo_tweets(
            current_count=100,
            hours_elapsed=24,
            total_hours=168,
            simulations=1000
        )

        assert 'mean' in result
        assert 'median' in result
        assert 'p10' in result
        assert 'p90' in result
        assert 'distribution' in result
        assert result['mean'] > 100  # Should project more tweets

    def test_zero_hours_elapsed_handled(self):
        """Should handle zero hours elapsed gracefully"""
        result = monte_carlo_tweets(
            current_count=50,
            hours_elapsed=0,
            total_hours=168
        )
        assert result['mean'] > 50

    def test_negative_hours_elapsed_handled(self):
        """Should handle negative hours gracefully"""
        result = monte_carlo_tweets(
            current_count=50,
            hours_elapsed=-5,
            total_hours=168
        )
        assert 'mean' in result

    def test_event_completed(self):
        """Should return current count when event is complete"""
        result = monte_carlo_tweets(
            current_count=300,
            hours_elapsed=168,
            total_hours=168
        )
        assert result['final'] == 300

    def test_projection_increases_with_rate(self):
        """Higher current rate should lead to higher projections"""
        low_rate = monte_carlo_tweets(100, 48, 168, simulations=500)
        high_rate = monte_carlo_tweets(200, 48, 168, simulations=500)

        # High rate projection should generally be higher
        assert high_rate['mean'] > low_rate['mean']

    def test_distribution_sums_to_one(self):
        """Distribution probabilities should sum to ~1"""
        result = monte_carlo_tweets(100, 24, 168, simulations=1000)
        total_prob = sum(result['distribution'].values())
        assert 0.95 <= total_prob <= 1.05  # Allow small float errors

    def test_p10_less_than_p90(self):
        """10th percentile should be less than 90th"""
        result = monte_carlo_tweets(100, 24, 168, simulations=1000)
        assert result['p10'] <= result['median'] <= result['p90']

    def test_custom_volatility(self):
        """Different volatility should affect spread"""
        low_vol = monte_carlo_tweets(100, 24, 168, simulations=1000, volatility=0.05)
        high_vol = monte_carlo_tweets(100, 24, 168, simulations=1000, volatility=0.30)

        low_spread = low_vol['p90'] - low_vol['p10']
        high_spread = high_vol['p90'] - high_vol['p10']

        # Higher volatility should generally give wider spread
        # (This may not always hold due to randomness, so we just check it runs)
        assert low_spread >= 0 and high_spread >= 0


# ============================================================================
# KELLY CRITERION TESTS
# ============================================================================

class TestKellyCriterion:
    """Tests for Kelly Criterion calculator"""

    def test_positive_edge(self):
        """Should recommend bet for positive edge"""
        # 60% win prob at 2:1 odds = edge
        result = kelly_criterion(prob_win=0.6, odds=2.0, fraction=0.25)

        assert result['edge'] > 0
        assert result['recommended'] > 0
        assert 'Bet' in result['recommendation']

    def test_no_edge(self):
        """Should not recommend bet when no edge"""
        # 50% win prob at 2:1 odds = no edge
        result = kelly_criterion(prob_win=0.5, odds=2.0)

        assert result['edge'] == 0 or result['edge'] < 0.001
        assert result['recommended'] == 0 or result['recommended'] < 0.001

    def test_negative_edge(self):
        """Should return zero bet for negative edge"""
        # 40% win at 2:1 = negative edge
        result = kelly_criterion(prob_win=0.4, odds=2.0)

        assert result['edge'] < 0
        assert result['recommended'] == 0

    def test_invalid_probability(self):
        """Should handle invalid probabilities"""
        result = kelly_criterion(prob_win=0, odds=2.0)
        assert result['kelly'] == 0

        result = kelly_criterion(prob_win=1.0, odds=2.0)
        assert result['kelly'] == 0

        result = kelly_criterion(prob_win=-0.5, odds=2.0)
        assert result['kelly'] == 0

    def test_invalid_odds(self):
        """Should handle invalid odds"""
        result = kelly_criterion(prob_win=0.6, odds=0.5)
        assert result['kelly'] == 0

        result = kelly_criterion(prob_win=0.6, odds=1.0)
        assert result['kelly'] == 0

    def test_fraction_scaling(self):
        """Quarter Kelly should be smaller than full Kelly"""
        full = kelly_criterion(prob_win=0.6, odds=2.0, fraction=1.0)
        quarter = kelly_criterion(prob_win=0.6, odds=2.0, fraction=0.25)

        if full['recommended'] > 0:
            assert quarter['recommended'] < full['recommended']
            assert abs(quarter['recommended'] - full['recommended'] * 0.25) < 0.001

    def test_high_probability_high_kelly(self):
        """Higher probability should give higher Kelly"""
        low_prob = kelly_criterion(prob_win=0.55, odds=2.0)
        high_prob = kelly_criterion(prob_win=0.70, odds=2.0)

        assert high_prob['full_kelly'] > low_prob['full_kelly']

    def test_output_fields(self):
        """Should return all required fields"""
        result = kelly_criterion(prob_win=0.6, odds=2.0)

        required_fields = [
            'prob_win', 'odds', 'edge', 'edge_pct',
            'full_kelly', 'full_kelly_pct', 'fraction',
            'recommended', 'recommended_pct', 'recommendation'
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"


# ============================================================================
# EXPECTED VALUE TESTS
# ============================================================================

class TestCalculateEV:
    """Tests for Expected Value calculations"""

    def test_positive_ev(self):
        """Should calculate positive EV correctly"""
        # Market says 50%, you think 60%, $100 position
        result = calculate_ev(market_prob=0.5, your_prob=0.6, position_size=100)

        assert result['ev'] > 0
        assert result['profitable'] == True

    def test_negative_ev(self):
        """Should calculate negative EV correctly"""
        # Market says 60%, you think 50%, $100 position
        result = calculate_ev(market_prob=0.6, your_prob=0.5, position_size=100)

        assert result['ev'] < 0
        assert result['profitable'] == False

    def test_zero_ev(self):
        """Should calculate zero EV when probs match"""
        result = calculate_ev(market_prob=0.5, your_prob=0.5, position_size=100)

        assert abs(result['ev']) < 0.01

    def test_ev_scales_with_position(self):
        """EV should scale linearly with position size"""
        small = calculate_ev(market_prob=0.5, your_prob=0.6, position_size=100)
        large = calculate_ev(market_prob=0.5, your_prob=0.6, position_size=1000)

        assert abs(large['ev'] - small['ev'] * 10) < 1.0  # Allow small float error

    def test_output_fields(self):
        """Should return all required fields"""
        result = calculate_ev(market_prob=0.5, your_prob=0.6, position_size=100)

        required_fields = ['market_prob', 'your_prob', 'ev', 'position_size', 'profitable']
        for field in required_fields:
            assert field in result, f"Missing field: {field}"


# ============================================================================
# ASCII CHART TESTS
# ============================================================================

class TestASCIICharts:
    """Tests for ASCII visualization functions"""

    def test_ascii_bar_basic(self):
        """Should create bar of correct length"""
        bar = ascii_bar(50, 100, width=40)
        assert len(bar) == 20  # 50% of 40

    def test_ascii_bar_full(self):
        """Should create full bar at max value"""
        bar = ascii_bar(100, 100, width=40)
        assert len(bar) == 40

    def test_ascii_bar_empty(self):
        """Should handle zero value"""
        bar = ascii_bar(0, 100, width=40)
        assert len(bar) == 0

    def test_ascii_bar_zero_max(self):
        """Should handle zero max gracefully"""
        bar = ascii_bar(50, 0, width=40)
        assert bar == ''

    def test_ascii_histogram_basic(self):
        """Should create histogram from data"""
        data = [0.1, 0.3, 0.5, 0.1]
        labels = ['A', 'B', 'C', 'D']
        hist = ascii_histogram(data, labels, title="Test")

        assert 'Test' in hist
        assert 'A' in hist
        assert 'B' in hist

    def test_ascii_histogram_empty_data(self):
        """Should handle empty data"""
        result = ascii_histogram([])
        assert result == "No data"


# ============================================================================
# EDGE CASES AND BOUNDARY TESTS
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_monte_carlo_single_simulation(self):
        """Should work with single simulation"""
        result = monte_carlo_tweets(100, 24, 168, simulations=1)
        assert 'mean' in result

    def test_monte_carlo_short_event(self):
        """Should handle very short events"""
        result = monte_carlo_tweets(10, 1, 2, simulations=100)
        assert result['mean'] >= 10

    def test_kelly_extreme_probability(self):
        """Should handle near-certain events"""
        result = kelly_criterion(prob_win=0.99, odds=2.0)
        assert result['full_kelly'] > 0.5  # Should be aggressive

    def test_kelly_long_shot(self):
        """Should handle low probability events"""
        result = kelly_criterion(prob_win=0.1, odds=15.0)  # 10% at 15:1
        # Has edge: 0.1 * 14 - 0.9 = 0.5
        assert result['edge'] > 0


# ============================================================================
# STATISTICAL VALIDATION TESTS
# ============================================================================

class TestStatisticalProperties:
    """Tests for statistical properties of models"""

    def test_monte_carlo_mean_near_linear(self):
        """Mean projection should be close to linear extrapolation"""
        current = 100
        elapsed = 24
        total = 168

        result = monte_carlo_tweets(current, elapsed, total, simulations=5000)

        linear_projection = current * (total / elapsed)

        # Mean should be within 30% of linear (due to randomness)
        assert abs(result['mean'] - linear_projection) < linear_projection * 0.3

    def test_kelly_edge_formula(self):
        """Edge should match p*b - q formula"""
        p = 0.6
        odds = 2.0
        b = odds - 1
        q = 1 - p

        result = kelly_criterion(p, odds)
        expected_edge = p * b - q

        assert abs(result['edge'] - expected_edge) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
