"""Tests for the shared histogram bin-edge resolution."""

from __future__ import annotations

import numpy as np
import pytest

from k4bench.analysis.plots._binning import MAX_BINS, describe_binning, resolve_bin_edges


@pytest.fixture
def sample():
    rng = np.random.default_rng(0)
    return rng.normal(10.0, 1.0, 500)


class TestResolveBinEdges:

    def test_auto_matches_numpy(self, sample):
        assert np.array_equal(
            resolve_bin_edges(sample), np.histogram_bin_edges(sample, bins="auto")
        )

    def test_int_bins_matches_numpy(self, sample):
        edges = resolve_bin_edges(sample, bins=17)
        assert len(edges) - 1 == 17
        assert np.array_equal(edges, np.histogram_bin_edges(sample, bins=17))

    def test_explicit_edges_pass_through(self, sample):
        explicit = np.linspace(5.0, 15.0, 11)
        assert np.array_equal(resolve_bin_edges(sample, bins=explicit), explicit)

    @pytest.mark.parametrize(
        ("edges", "message"),
        [
            ([], "at least two"),
            ([1.0], "at least two"),
            ([1.0, 1.0], "strictly increasing"),
            ([2.0, 1.0], "strictly increasing"),
            ([1.0, np.nan, 2.0], "finite"),
            ([1.0, np.inf], "finite"),
        ],
    )
    def test_malformed_explicit_edges_rejected(self, sample, edges, message):
        with pytest.raises(ValueError, match=message):
            resolve_bin_edges(sample, bins=edges)

    def test_bin_width_gives_uniform_bins(self, sample):
        edges = resolve_bin_edges(sample, bin_width=0.25)
        assert np.allclose(np.diff(edges), 0.25)

    def test_bin_width_covers_the_maximum(self, sample):
        """The last edge must sit past the data maximum, or entries are dropped."""
        edges = resolve_bin_edges(sample, bin_width=0.3)
        assert edges[0] == pytest.approx(sample.min())
        assert edges[-1] >= sample.max()
        counts, _ = np.histogram(sample, bins=edges)
        assert counts.sum() == len(sample)

    def test_bin_width_larger_than_span_gives_one_bin(self):
        edges = resolve_bin_edges(np.array([1.0, 1.5, 2.0]), bin_width=100.0)
        assert len(edges) - 1 == 1

    def test_single_valued_data_gives_one_bin(self):
        edges = resolve_bin_edges(np.full(10, 3.0), bin_width=0.5)
        assert len(edges) - 1 == 1
        counts, _ = np.histogram(np.full(10, 3.0), bins=edges)
        assert counts.sum() == 10

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan")])
    def test_non_positive_bin_width_rejected(self, sample, bad):
        with pytest.raises(ValueError, match="bin_width must be a positive"):
            resolve_bin_edges(sample, bin_width=bad)

    def test_bins_and_bin_width_together_rejected(self, sample):
        with pytest.raises(ValueError, match="not both"):
            resolve_bin_edges(sample, bins=20, bin_width=0.5)

    def test_too_many_bins_rejected(self, sample):
        with pytest.raises(ValueError, match=f"limit of {MAX_BINS}"):
            resolve_bin_edges(sample, bins=MAX_BINS + 1)

    def test_integer_limit_is_checked_before_numpy_allocates(self, sample, monkeypatch):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("NumPy allocation was attempted")

        monkeypatch.setattr(np, "histogram_bin_edges", fail_if_called)
        with pytest.raises(ValueError, match=f"limit of {MAX_BINS}"):
            resolve_bin_edges(sample, bins=MAX_BINS + 1)

    def test_too_narrow_bin_width_rejected(self, sample):
        span = sample.max() - sample.min()
        with pytest.raises(ValueError, match=f"limit of {MAX_BINS}"):
            resolve_bin_edges(sample, bin_width=span / (MAX_BINS * 2))

    def test_width_limit_is_checked_before_edge_array_allocation(self, sample, monkeypatch):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("edge allocation was attempted")

        monkeypatch.setattr(np, "arange", fail_if_called)
        span = sample.max() - sample.min()
        with pytest.raises(ValueError, match=f"limit of {MAX_BINS}"):
            resolve_bin_edges(sample, bin_width=span / (MAX_BINS * 2))

    def test_empty_data_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            resolve_bin_edges(np.array([]))

    def test_non_finite_data_rejected(self):
        with pytest.raises(ValueError, match="NaN or infinite"):
            resolve_bin_edges(np.array([1.0, np.nan, 3.0]))


class TestDescribeBinning:

    def test_uniform_edges(self):
        info = describe_binning(np.linspace(0.0, 10.0, 11))
        assert info["n_bins"] == 10
        assert info["bin_width"] == pytest.approx(1.0)
        assert info["bin_range"] == (0.0, 10.0)
        assert info["uniform"] is True

    def test_non_uniform_edges_flagged(self):
        info = describe_binning(np.array([0.0, 1.0, 5.0, 6.0]))
        assert info["uniform"] is False
        assert info["bin_width"] == pytest.approx(2.0)

    @pytest.mark.parametrize("edges", [[], [1.0], [1.0, 1.0], [1.0, np.nan]])
    def test_invalid_edges_rejected(self, edges):
        with pytest.raises(ValueError, match="at least two|finite|strictly increasing"):
            describe_binning(np.asarray(edges))
