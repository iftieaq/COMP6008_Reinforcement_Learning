"""
Day profiles and uncertainty.

A day profile holds everything random about one simulated day: solar and load
for each timestep, when the EV arrives and leaves, its charge on arrival, and
the home battery's charge at midnight. The environment reads one profile per
episode, so all randomness in the environment comes from here.

Initial version by Person 1 so the environment can run. Person 3 owns the
final profile shapes and uncertainty values.
"""

import math
from dataclasses import dataclass

import numpy as np

from config import (
    EV_TARGET_SOC,
    NOMINAL_EV_ARRIVAL_STEP,
    NOMINAL_EV_DEPARTURE_STEP,
    NOMINAL_LOAD_PERIODS,
    NUMBER_OF_TEST_DAYS,
    SOLAR_PEAK_OUTPUT_KW,
    STEP_HOURS,
    STEPS_PER_DAY,
    SUNRISE_HOUR,
    SUNSET_HOUR,
    TEST_SEED_START,
    UNCERTAINTY_LEVELS,
)


@dataclass
class DayProfile:
    """
    The random inputs for one simulated day.

    Attributes:
        solar_kwh: array of 48 values, solar energy produced in each timestep.
        load_kwh: array of 48 values, household energy used in each timestep.
        ev_arrival_step: first timestep the EV is plugged in.
        ev_departure_step: first timestep the EV is gone (plugged in for
            steps ev_arrival_step to ev_departure_step - 1).
        ev_arrival_soc: EV state of charge (0 to 1) when it arrives.
        ev_target_soc: EV state of charge required at departure.
        initial_battery_soc: home battery state of charge at 00:00.
    """

    solar_kwh: np.ndarray
    load_kwh: np.ndarray
    ev_arrival_step: int
    ev_departure_step: int
    ev_arrival_soc: float
    ev_target_soc: float
    initial_battery_soc: float


def step_midpoint_hour(time_step):
    """
    Return the clock hour at the middle of a timestep.

    Args:
        time_step: timestep index, 0 to 47.

    Returns:
        Hour of day as a float, for example 12.25 for timestep 24.
    """
    return (time_step + 0.5) * STEP_HOURS


def nominal_solar_profile_kwh():
    """
    Build the average clear-day solar profile.

    Output follows a half sine curve between sunrise and sunset, peaking at
    SOLAR_PEAK_OUTPUT_KW around noon, and is zero at night.

    Returns:
        Array of 48 values, solar energy in kWh for each timestep.
    """
    solar_kwh = np.zeros(STEPS_PER_DAY)
    daylight_hours = SUNSET_HOUR - SUNRISE_HOUR

    for time_step in range(STEPS_PER_DAY):
        hour = step_midpoint_hour(time_step)
        if SUNRISE_HOUR < hour < SUNSET_HOUR:
            fraction_of_daylight = (hour - SUNRISE_HOUR) / daylight_hours
            power_kw = SOLAR_PEAK_OUTPUT_KW * math.sin(math.pi * fraction_of_daylight)
            solar_kwh[time_step] = power_kw * STEP_HOURS

    return solar_kwh


def nominal_load_profile_kwh():
    """
    Build the average household load profile from NOMINAL_LOAD_PERIODS.

    Returns:
        Array of 48 values, household energy use in kWh for each timestep.
    """
    load_kwh = np.zeros(STEPS_PER_DAY)

    for time_step in range(STEPS_PER_DAY):
        hour = step_midpoint_hour(time_step)
        for start_hour, end_hour, power_kw in NOMINAL_LOAD_PERIODS:
            if start_hour <= hour < end_hour:
                load_kwh[time_step] = power_kw * STEP_HOURS

    return load_kwh


def add_noise(nominal_values, noise_std, random_generator):
    """
    Multiply each value by (1 + noise), where noise is normally distributed.

    Args:
        nominal_values: array of values to perturb.
        noise_std: standard deviation of the noise (0.1 means about 10%).
        random_generator: numpy random Generator.

    Returns:
        New array of the same length, with negative values clipped to 0.
    """
    noise = random_generator.normal(0.0, noise_std, size=len(nominal_values))
    return np.clip(nominal_values * (1.0 + noise), 0.0, None)


def generate_day_profile(uncertainty_level, random_generator):
    """
    Generate one random day for the given uncertainty level.

    Args:
        uncertainty_level: "low", "medium" or "high" (keys of UNCERTAINTY_LEVELS).
        random_generator: numpy random Generator that drives all randomness.

    Returns:
        A DayProfile.
    """
    settings = UNCERTAINTY_LEVELS[uncertainty_level]

    weather_factor = random_generator.uniform(*settings["weather_factor_range"])
    cloudy_solar_kwh = nominal_solar_profile_kwh() * weather_factor
    solar_kwh = add_noise(cloudy_solar_kwh, settings["solar_noise_std"], random_generator)

    load_kwh = add_noise(nominal_load_profile_kwh(), settings["load_noise_std"], random_generator)

    jitter = settings["ev_time_jitter_steps"]
    ev_arrival_step = NOMINAL_EV_ARRIVAL_STEP + int(random_generator.integers(-jitter, jitter + 1))
    ev_departure_step = NOMINAL_EV_DEPARTURE_STEP + int(random_generator.integers(-jitter, jitter + 1))

    ev_arrival_soc = random_generator.uniform(*settings["ev_arrival_soc_range"])
    initial_battery_soc = random_generator.uniform(*settings["initial_battery_soc_range"])

    return DayProfile(
        solar_kwh=solar_kwh,
        load_kwh=load_kwh,
        ev_arrival_step=ev_arrival_step,
        ev_departure_step=ev_departure_step,
        ev_arrival_soc=float(ev_arrival_soc),
        ev_target_soc=EV_TARGET_SOC,
        initial_battery_soc=float(initial_battery_soc),
    )


def generate_test_profiles(uncertainty_level, number_of_days=NUMBER_OF_TEST_DAYS):
    """
    Generate the fixed test days for one uncertainty level.

    Each test day uses its own seed starting at TEST_SEED_START, so the test
    set is identical on every run and never overlaps with training.

    Args:
        uncertainty_level: "low", "medium" or "high".
        number_of_days: how many test days to generate.

    Returns:
        List of DayProfile objects.
    """
    test_profiles = []
    for day_index in range(number_of_days):
        random_generator = np.random.default_rng(TEST_SEED_START + day_index)
        test_profiles.append(generate_day_profile(uncertainty_level, random_generator))
    return test_profiles
