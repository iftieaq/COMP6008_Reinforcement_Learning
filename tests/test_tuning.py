"""
Checks for the hyperparameter tuning helpers.
"""

from q_learning import QLearningAgent
from tuning import (
    choose_best_settings,
    generate_validation_profiles,
    one_at_a_time_configurations,
    run_configuration,
    summarise_learning_curve,
)

DEFAULTS = {"learning_rate": 0.1, "discount_factor": 0.99}
CANDIDATES = {"learning_rate": [0.05, 0.1, 0.2], "discount_factor": [0.9, 0.99]}


def test_one_at_a_time_changes_one_setting_per_configuration():
    configurations = one_at_a_time_configurations(DEFAULTS, CANDIDATES)
    names = [name for name, settings in configurations]
    assert names == ["default", "learning_rate=0.05", "learning_rate=0.2", "discount_factor=0.9"]

    for name, settings in configurations[1:]:
        changed = [key for key in DEFAULTS if settings[key] != DEFAULTS[key]]
        assert len(changed) == 1


def test_best_value_is_chosen_per_setting():
    summaries = [
        {"name": "default", "settings": DEFAULTS, "mean_reward": -3.0},
        {"name": "learning_rate=0.05", "settings": {**DEFAULTS, "learning_rate": 0.05}, "mean_reward": -2.5},
        {"name": "learning_rate=0.2", "settings": {**DEFAULTS, "learning_rate": 0.2}, "mean_reward": -3.5},
        {"name": "discount_factor=0.9", "settings": {**DEFAULTS, "discount_factor": 0.9}, "mean_reward": -4.0},
    ]
    best = choose_best_settings(summaries, DEFAULTS, CANDIDATES)
    assert best == {"learning_rate": 0.05, "discount_factor": 0.99}


def test_learning_curve_is_shrunk_to_block_averages():
    assert summarise_learning_curve([1, 1, 3, 3], number_of_points=2) == [1.0, 3.0]


def test_validation_days_differ_from_test_days():
    from scenarios import generate_test_profiles
    validation_day = generate_validation_profiles("medium", 1)[0]
    test_day = generate_test_profiles("medium", 1)[0]
    assert validation_day.ev_arrival_soc != test_day.ev_arrival_soc


def test_short_run_produces_expected_fields():
    settings = {"learning_rate": 0.1, "discount_factor": 0.99, "epsilon_start": 1.0,
                "epsilon_decay": 0.99, "epsilon_min": 0.01}
    result = run_configuration(QLearningAgent, settings, seed=0, uncertainty_level="low",
                               number_of_episodes=200,
                               validation_profiles=generate_validation_profiles("low", 5))
    assert result["seed"] == 0
    assert len(result["learning_curve"]) == 100
    assert result["states_visited"] > 0
    assert "total_reward" in result
