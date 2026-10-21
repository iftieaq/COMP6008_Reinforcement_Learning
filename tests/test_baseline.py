"""
Checks for the rule-based baseline.
"""

from baseline import RuleBasedController
from config import ACTION_TABLE, BATTERY_CHARGE
from environment import HouseholdEnergyEnv
from scenarios import generate_test_profiles
from training import average_results, evaluate_controller


def test_baseline_returns_valid_actions():
    env = HouseholdEnergyEnv("high")
    controller = RuleBasedController()
    observation, info = env.reset(seed=0)
    done = False
    while not done:
        action = controller.act(observation)
        assert action in ACTION_TABLE
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated


def test_baseline_charges_battery_from_solar_surplus():
    observation = {
        "battery_soc": 0.5,
        "ev_soc": 0.0,
        "solar_kwh": 2.0,
        "load_kwh": 0.3,
        "time_step": 24,
        "ev_plugged_in": 0,
        "steps_to_departure": 0,
    }
    battery_command, charge_ev = ACTION_TABLE[RuleBasedController().act(observation)]
    assert battery_command == BATTERY_CHARGE
    assert charge_ev is False


def test_baseline_meets_ev_target_on_low_uncertainty_days():
    env = HouseholdEnergyEnv("low")
    results = evaluate_controller(env, RuleBasedController(), generate_test_profiles("low", 50))
    assert average_results(results)["ev_target_met"] == 1.0
