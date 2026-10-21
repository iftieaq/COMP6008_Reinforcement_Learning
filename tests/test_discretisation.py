"""
Checks for turning observations into Q-table state keys.
"""

from config import (
    EV_STATUS_AWAY,
    EV_STATUS_CRITICAL,
    EV_STATUS_MODERATE,
    EV_STATUS_RELAXED,
    EV_STATUS_TARGET_MET,
    EV_STATUS_TIGHT,
)
from environment import discretise_observation, ev_steps_needed_to_reach_target, find_bin, find_ev_status


def make_observation(ev_plugged_in=1, ev_soc=0.5, steps_to_departure=10):
    """
    Build an observation dict with typical values, overriding the EV fields.

    Args:
        ev_plugged_in: 0 or 1.
        ev_soc: EV state of charge.
        steps_to_departure: steps the EV can still charge.

    Returns:
        Observation dict.
    """
    return {
        "battery_soc": 0.5,
        "ev_soc": ev_soc,
        "solar_kwh": 1.0,
        "load_kwh": 0.25,
        "time_step": 24,
        "ev_plugged_in": ev_plugged_in,
        "steps_to_departure": steps_to_departure,
    }


def test_find_bin_places_values_correctly():
    edges = [0.2, 0.4, 0.6, 0.8]
    assert find_bin(0.0, edges) == 0
    assert find_bin(0.19, edges) == 0
    assert find_bin(0.2, edges) == 1
    assert find_bin(0.45, edges) == 2
    assert find_bin(1.0, edges) == 4


def test_steps_needed_to_reach_target():
    assert ev_steps_needed_to_reach_target(0.8) == 0
    assert ev_steps_needed_to_reach_target(0.9) == 0
    # 30% of 60 kWh = 18 kWh, at 3.5 kWh per step = 5.14, rounded up to 6
    assert ev_steps_needed_to_reach_target(0.5) == 6


def test_ev_status_covers_every_case():
    assert find_ev_status(make_observation(ev_plugged_in=0)) == EV_STATUS_AWAY
    assert find_ev_status(make_observation(ev_soc=0.85)) == EV_STATUS_TARGET_MET
    # ev_soc 0.5 needs 6 steps, so slack = steps_to_departure - 6
    assert find_ev_status(make_observation(steps_to_departure=14)) == EV_STATUS_RELAXED
    assert find_ev_status(make_observation(steps_to_departure=10)) == EV_STATUS_MODERATE
    assert find_ev_status(make_observation(steps_to_departure=8)) == EV_STATUS_TIGHT
    assert find_ev_status(make_observation(steps_to_departure=6)) == EV_STATUS_CRITICAL
    assert find_ev_status(make_observation(steps_to_departure=2)) == EV_STATUS_CRITICAL


def test_state_key_has_expected_shape():
    state = discretise_observation(make_observation())
    # (battery_bin, solar_bin, load_bin, time_step, ev_status)
    assert state == (5, 2, 0, 24, EV_STATUS_MODERATE)
