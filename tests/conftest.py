"""pytest 共享 fixtures 和配置。"""
import sys
import os
import pytest

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_queries():
    return [
        {"id": "Q1", "query": "What is the calcium content of limestone?",
         "category": "Feed Ingredient Bank", "type": "single-value",
         "expected_value": 38.0},
        {"id": "Q2", "query": "Compare CP of corn and wheat",
         "category": "Feed Ingredient Bank", "type": "comparison"},
        {"id": "Q3", "query": "What is FCR?",
         "category": "Definitions", "type": "definition"},
        {"id": "Q4", "query": "What is the BW of Ross 308 at 35 days?",
         "category": "Performance Management", "type": "single-value"},
    ]


@pytest.fixture
def sample_responses():
    return [
        "The calcium content of limestone is approximately 38%.",
        "Corn has 8.5% CP while wheat has 11.5% CP.",
        "FCR stands for Feed Conversion Ratio.",
        "The body weight of Ross 308 at 35 days is about 2100g.",
    ]
