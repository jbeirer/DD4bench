"""Key4hep stack provenance — the upstream commit each package was built from.

A benchmark measures a *stack*, not just a detector: when a metric steps, the
cause is almost always a commit that landed in one of the ~60 packages the
nightly builds from git. Recording those commits at benchmark time is what
later lets a regression be traced back to the pull requests that could have
caused it.

Capture has to happen while the benchmark runs. LCG rotates its weekday views,
so a stack that is not recorded when it ran is soon unrecoverable — this package is called from
``nightly_benchmark.sh`` rather than reconstructed after the fact.

- :mod:`k4bench.provenance.stack` — read an LCG view's manifest off CVMFS.
"""
