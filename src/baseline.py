"""
Rule-based baseline controller.

A fixed, hand-written policy used as the reference point for the RL agents.
It follows the rules in docs/mdp-and-interfaces.md, section 10: use solar
first, store surplus in the battery before the EV, use the battery before
the grid, and charge the EV whenever its deadline becomes urgent.

These rules are frozen once agreed and must not be changed after seeing
results.
"""

from config import (
    BATTERY_CHARGE,
    BATTERY_DISCHARGE,
    BATTERY_IDLE,
    EV_MAX_ENERGY_PER_STEP_KWH,
    EV_TARGET_SOC,
)
from environment import ev_steps_needed_to_reach_target, find_action_id

BATTERY_FULL_SOC = 0.99
BATTERY_EMPTY_SOC = 0.01


class RuleBasedController:
    """
    Picks an action from the raw observation using fixed priority rules.

    Provides act(observation), the same method the evaluation code calls for
    every controller.
    """

    def act(self, observation):
        """
        Choose an action for the current step.

        Args:
            observation: observation dict from HouseholdEnergyEnv.

        Returns:
            Action ID from 0 to 5.
        """
        charge_ev = self.should_charge_ev(observation)
        battery_command = self.choose_battery_command(observation, charge_ev)
        return find_action_id(battery_command, charge_ev)

    def should_charge_ev(self, observation):
        """
        Decide whether to charge the EV this step.

        Charge if the EV is plugged in, below target, and either the deadline
        is urgent or there is solar surplus that the full battery cannot take.

        Args:
            observation: observation dict.

        Returns:
            True to charge the EV.
        """
        if observation["ev_plugged_in"] == 0:
            return False
        if observation["ev_soc"] >= EV_TARGET_SOC:
            return False

        steps_needed = ev_steps_needed_to_reach_target(observation["ev_soc"])
        deadline_is_urgent = observation["steps_to_departure"] <= steps_needed + 1

        has_solar_surplus = observation["solar_kwh"] > observation["load_kwh"]
        battery_is_full = observation["battery_soc"] >= BATTERY_FULL_SOC

        return deadline_is_urgent or (has_solar_surplus and battery_is_full)

    def choose_battery_command(self, observation, charge_ev):
        """
        Decide what the home battery does this step.

        Charge on solar surplus, discharge if the house needs energy that solar
        does not cover, otherwise stay idle.

        Args:
            observation: observation dict.
            charge_ev: whether the EV is charging this step.

        Returns:
            BATTERY_CHARGE, BATTERY_DISCHARGE or BATTERY_IDLE.
        """
        solar_kwh = observation["solar_kwh"]
        load_kwh = observation["load_kwh"]
        battery_soc = observation["battery_soc"]

        ev_demand_kwh = EV_MAX_ENERGY_PER_STEP_KWH if charge_ev else 0.0
        has_solar_surplus = solar_kwh > load_kwh
        house_needs_energy = load_kwh + ev_demand_kwh > solar_kwh

        if has_solar_surplus and battery_soc < BATTERY_FULL_SOC:
            return BATTERY_CHARGE
        if house_needs_energy and battery_soc > BATTERY_EMPTY_SOC:
            return BATTERY_DISCHARGE
        return BATTERY_IDLE
