"""
flow_engineering/__init__.py
─────────────────────────────
Flow Engineering module for FITGEN.AI — Assignment 5.

Provides LangChain-based prompt flow experiments, chain variants,
Azure Prompt Flow integration, iterative improvement tracking,
and automated evaluation.
"""

from flow_engineering.chain_variants import build_all_variants
from flow_engineering.prompt_flow import FlowRunner
from flow_engineering.flow_evaluator import FlowEvaluator
from flow_engineering.iteration_tracker import IterationTracker

__all__ = [
    "build_all_variants",
    "FlowRunner",
    "FlowEvaluator",
    "IterationTracker",
]
