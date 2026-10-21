"""
Sanity checks for the household environment: episode length, physical
limits, energy balance and reward bookkeeping.
"""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from config import (
    BATTERY_MAX_ENERGY_PER_STEP_KWH,
    EV_CAPACITY_KWH,
    EV_MAX_ENERGY_PER_STEP_KWH,
    NUMBER_OF_ACTIONS,
    STEPS_PER_DAY,
)
from environment import HouseholdEnergyEnv

TOLERANCE = 1e-9


def run_random_day(uncertainty_level, seed):
    """
    Run one day with random actions.

    Args:
        uncertainty_level: "low", "medium" or "high".
        seed: seed for the day and the random actions.

    Returns:
        Tuple (env, list of (observation_before, action, reward, info) per step).
    """
    env = HouseholdEnergyEnv(uncertainty_level)
    observation, info = env.reset(seed=seed)
    action_generator = np.random.default_rng(seed)

    steps = []
    done = False
    while not done:
        action = int(action_generator.integers(NUMBER_OF_ACTIONS))
        next_observation, reward, terminated, truncated, info = env.step(action)
        steps.append((observation, action, reward, info))
        observation = next_observation
        done = terminated or truncated
    return env, steps


@pytest.mark.parametrize("uncertainty_level", ["low", "medium", "high"])
def test_episode_lasts_one_day(uncertainty_level):
    env, steps = run_random_day(uncertainty_level, seed=1)
    assert len(steps) == STEPS_PER_DAY


def test_same_seed_gives_same_day():
    first_env = HouseholdEnergyEnv("high")
    second_env = HouseholdEnergyEnv("high")
    first_env.reset(seed=42)
    second_env.reset(seed=42)
    assert np.array_equal(first_env.profile.solar_kwh, second_env.profile.solar_kwh)
    assert first_env.profile.ev_arrival_step == second_env.profile.ev_arrival_step


@pytest.mark.parametrize("seed", range(20))
def test_energy_balance_holds_every_step(seed):
    env, steps = run_random_day("high", seed)
    for observation, action, reward, info in steps:
        energy_in = info["solar_kwh"] + info["battery_discharge_kwh"] + info["grid_import_kwh"]
        energy_out = (info["load_kwh"] + info["ev_charge_kwh"]
                      + info["battery_charge_kwh"] + info["grid_export_kwh"])
        assert energy_in == pytest.approx(energy_out, abs=TOLERANCE)


@pytest.mark.parametrize("seed", range(20))
def test_physical_limits_are_respected(seed):
    env, steps = run_random_day("high", seed)
    for observation, action, reward, info in steps:
        assert 0.0 <= info["battery_soc"] <= 1.0 + TOLERANCE
        assert 0.0 <= info["ev_soc"] <= 1.0 + TOLERANCE
        assert info["battery_charge_kwh"] <= BATTERY_MAX_ENERGY_PER_STEP_KWH + TOLERANCE
        assert info["battery_discharge_kwh"] <= BATTERY_MAX_ENERGY_PER_STEP_KWH + TOLERANCE
        assert info["ev_charge_kwh"] <= EV_MAX_ENERGY_PER_STEP_KWH + TOLERANCE
        assert info["battery_charge_kwh"] * info["battery_discharge_kwh"] == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_ev_only_charges_when_plugged_in(seed):
    env, steps = run_random_day("high", seed)
    for observation, action, reward, info in steps:
        if observation["ev_plugged_in"] == 0:
            assert info["ev_charge_kwh"] == 0.0


@pytest.mark.parametrize("seed", range(20))
def test_battery_never_exports_to_grid(seed):
    env, steps = run_random_day("high", seed)
    for observation, action, reward, info in steps:
        house_demand_kwh = info["load_kwh"] + info["ev_charge_kwh"] - info["solar_kwh"]
        assert info["battery_discharge_kwh"] <= max(0.0, house_demand_kwh) + TOLERANCE


@pytest.mark.parametrize("seed", range(20))
def test_reward_parts_add_up(seed):
    env, steps = run_random_day("medium", seed)
    for observation, action, reward, info in steps:
        reward_parts = info["reward_grid"] + info["reward_wear"] + info["reward_ev"] + info["reward_peak"]
        assert reward == pytest.approx(reward_parts, abs=TOLERANCE)


@pytest.mark.parametrize("seed", range(20))
def test_ev_shortfall_only_at_departure(seed):
    env, steps = run_random_day("medium", seed)
    last_plugged_in_step = env.profile.ev_departure_step - 1
    for observation, action, reward, info in steps:
        if info["time_step"] != last_plugged_in_step:
            assert info["ev_shortfall_kwh"] == 0.0


def test_ev_with_no_charging_gets_full_shortfall():
    env = HouseholdEnergyEnv("low")
    env.reset(seed=3)
    total_shortfall = 0.0
    for time_step in range(STEPS_PER_DAY):
        observation, reward, terminated, truncated, info = env.step(0)
        total_shortfall += info["ev_shortfall_kwh"]

    profile = env.profile
    expected_shortfall = (profile.ev_target_soc - profile.ev_arrival_soc) * EV_CAPACITY_KWH
    assert total_shortfall == pytest.approx(expected_shortfall)


def test_passes_gymnasium_environment_checker():
    with warnings.catch_warnings():
        # Gymnasium warns that plain Python floats are cast to numpy; this is harmless
        warnings.filterwarnings("ignore", message=".*Casting input x to numpy array.*")
        warnings.filterwarnings("ignore", message=".*Not able to test alternative render modes.*")
        # Solar and load have no fixed upper limit, so their maximum is infinity on purpose
        warnings.filterwarnings("ignore", message=".*maximum value is infinity.*")
        check_env(HouseholdEnergyEnv("medium"))


def test_test_days_differ_from_development_days():
    from scenarios import generate_development_profiles, generate_test_profiles
    development_day = generate_development_profiles("medium", 1)[0]
    test_day = generate_test_profiles("medium", 1)[0]
    assert development_day.ev_arrival_soc != test_day.ev_arrival_soc
