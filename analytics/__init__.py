"""Quantitative decision model: momentum, risk-adjusted efficiency, tail risk, implied sentiment.

Deliberately isolated from the trading path. server.py mounts this behind a
try/except, so an import error, a missing database or a numpy problem degrades
the dashboard to "no risk numbers" rather than taking order entry down with it.
"""
