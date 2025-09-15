"""Test utilities package for oligopoly tests.

This package provides shared utilities for testing, including database
management, test data generation, and common assertions.
"""

from .test_assertions import (
    assert_bertrand_cli_format,
    assert_bertrand_output_format,
    assert_consumer_surplus_calculation,
    assert_cournot_cli_format,
    assert_cournot_output_format,
    assert_event_log_structure,
    assert_hhi_calculation,
    assert_run_data_structure,
    assert_strategy_trajectory,
)
from .test_data import (
    create_sample_bertrand_cli_config,
    create_sample_bertrand_config,
    create_sample_collusion_config,
    create_sample_cournot_cli_config,
    create_sample_cournot_config,
    create_sample_experiment_config,
    create_sample_policy_config,
    create_sample_segmented_demand_config,
    create_sample_strategy_config,
)
from .test_db import (
    TestDatabaseManager,
    create_test_database,
    create_test_session,
    override_get_db_for_testing,
)

__all__ = [
    # Database utilities
    "TestDatabaseManager",
    "create_test_database",
    "create_test_session",
    "override_get_db_for_testing",
    # Test data utilities
    "create_sample_bertrand_config",
    "create_sample_cournot_config",
    "create_sample_bertrand_cli_config",
    "create_sample_cournot_cli_config",
    "create_sample_strategy_config",
    "create_sample_experiment_config",
    "create_sample_policy_config",
    "create_sample_segmented_demand_config",
    "create_sample_collusion_config",
    # Assertion utilities
    "assert_bertrand_output_format",
    "assert_cournot_output_format",
    "assert_bertrand_cli_format",
    "assert_cournot_cli_format",
    "assert_hhi_calculation",
    "assert_consumer_surplus_calculation",
    "assert_strategy_trajectory",
    "assert_run_data_structure",
    "assert_event_log_structure",
]
