"""
Household energy environment and state discretisation.

HouseholdEnergyEnv simulates one day of a house with rooftop solar, a home
battery, household load, an EV and the grid, in 48 half-hour steps. It follows
the gymnasium.Env interface: reset() starts a new day and step(action)
advances 30 minutes. The equations are in docs/mdp-and-interfaces.md,
sections 4 and 5.

discretise_observation() turns the environment's real-valued observation into
the tuple of bin numbers the tabular agents use as their state.
"""

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config import (
    ACTION_TABLE,
    BATTERY_CAPACITY_KWH,
    BATTERY_CHARGE,
    BATTERY_CHARGE_EFFICIENCY,
    BATTERY_DISCHARGE,
    BATTERY_MAX_ENERGY_PER_STEP_KWH,
    BATTERY_SOC_BIN_EDGES,
    BATTERY_WEAR_WEIGHT,
    EV_CAPACITY_KWH,
    EV_MAX_ENERGY_PER_STEP_KWH,
    EV_MODERATE_MIN_SLACK,
    EV_RELAXED_MIN_SLACK,
    EV_SHORTFALL_WEIGHT,
    EV_STATUS_AWAY,
    EV_STATUS_CRITICAL,
    EV_STATUS_MODERATE,
    EV_STATUS_RELAXED,
    EV_STATUS_TARGET_MET,
    EV_STATUS_TIGHT,
    EV_TARGET_SOC,
    EV_TIGHT_MIN_SLACK,
    EXPORT_PRICE_PER_KWH,
    IMPORT_PRICE_PER_KWH,
    LEFTOVER_BATTERY_VALUE_PER_KWH,
    LOAD_BIN_EDGES_KWH,
    NUMBER_OF_ACTIONS,
    PEAK_IMPORT_THRESHOLD_KWH,
    PEAK_IMPORT_WEIGHT,
    SOLAR_BIN_EDGES_KWH,
    STEPS_PER_DAY,
    UNCERTAINTY_LEVELS,
)
from scenarios import generate_day_profile


class HouseholdEnergyEnv(gym.Env):
    """
    One simulated day of household energy management.

    Each step the agent picks one of six actions (see ACTION_TABLE in
    config.py) that sets what the home battery does and whether the EV
    charges. The environment applies the physics, settles the grid balance
    and returns a reward equal to the negative cost of the step.

    Args:
        uncertainty_level: "low", "medium" or "high". Sets how much solar,
            load and the EV schedule vary from day to day.
    """

    metadata = {"render_modes": []}

    def __init__(self, uncertainty_level="medium"):
        if uncertainty_level not in UNCERTAINTY_LEVELS:
            raise ValueError(f"Unknown uncertainty level: {uncertainty_level}")

        self.uncertainty_level = uncertainty_level

        self.action_space = spaces.Discrete(NUMBER_OF_ACTIONS)
        self.observation_space = spaces.Dict({
            "battery_soc": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float64),
            "ev_soc": spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float64),
            "solar_kwh": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float64),
            "load_kwh": spaces.Box(low=0.0, high=np.inf, shape=(), dtype=np.float64),
            "time_step": spaces.Discrete(STEPS_PER_DAY + 1),
            "ev_plugged_in": spaces.Discrete(2),
            "steps_to_departure": spaces.Discrete(STEPS_PER_DAY + 1),
        })

        self.profile = None
        self.time_step = 0
        self.battery_energy_kwh = 0.0
        self.ev_energy_kwh = 0.0

    # -----------------------------------------------------------------------
    # Gymnasium interface
    # -----------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """
        Start a new day.

        Args:
            seed: optional random seed. Pass it once at the start of a run;
                later resets continue the same random sequence.
            options: optional dict. {"profile": day_profile} replays that exact
                day instead of generating a new one (used for test days).

        Returns:
            observation: dict of the starting state (see _get_observation).
            info: dict containing the day profile used.
        """
        super().reset(seed=seed)

        if options is not None and "profile" in options:
            self.profile = options["profile"]
        else:
            self.profile = generate_day_profile(self.uncertainty_level, self.np_random)

        self.time_step = 0
        self.battery_energy_kwh = self.profile.initial_battery_soc * BATTERY_CAPACITY_KWH
        self.ev_energy_kwh = self.profile.ev_arrival_soc * EV_CAPACITY_KWH

        return self._get_observation(), {"profile": self.profile}

    def step(self, action):
        """
        Apply one action and advance the simulation by 30 minutes.

        Args:
            action: integer action ID from 0 to 5.

        Returns:
            observation: dict of the next state.
            reward: negative cost of this step (float).
            terminated: True after the last timestep of the day.
            truncated: always False (episodes are never cut short).
            info: dict with energy flows and reward parts for analysis.
        """
        battery_command, charge_ev_requested = ACTION_TABLE[int(action)]

        time_step = self.time_step
        solar_kwh = float(self.profile.solar_kwh[time_step])
        load_kwh = float(self.profile.load_kwh[time_step])
        ev_plugged_in = self._is_ev_plugged_in(time_step)

        # Physics, in the order of the spec: EV first, then the battery
        ev_charge_kwh = self._charge_ev(charge_ev_requested and ev_plugged_in)

        battery_charge_kwh = 0.0
        battery_discharge_kwh = 0.0
        if battery_command == BATTERY_CHARGE:
            battery_charge_kwh = self._charge_battery()
        elif battery_command == BATTERY_DISCHARGE:
            house_demand_kwh = load_kwh + ev_charge_kwh - solar_kwh
            battery_discharge_kwh = self._discharge_battery(house_demand_kwh)

        # Grid balance: positive means buying, negative means selling
        net_grid_kwh = load_kwh + ev_charge_kwh + battery_charge_kwh - solar_kwh - battery_discharge_kwh
        grid_import_kwh = max(0.0, net_grid_kwh)
        grid_export_kwh = max(0.0, -net_grid_kwh)

        ev_shortfall_kwh = self._ev_shortfall_at_departure(time_step)
        peak_excess_kwh = max(0.0, grid_import_kwh - PEAK_IMPORT_THRESHOLD_KWH)

        grid_cost = (grid_import_kwh * IMPORT_PRICE_PER_KWH[time_step]
                     - grid_export_kwh * EXPORT_PRICE_PER_KWH[time_step])
        reward_grid = -grid_cost
        reward_wear = -BATTERY_WEAR_WEIGHT * (battery_charge_kwh + battery_discharge_kwh)
        reward_ev = -EV_SHORTFALL_WEIGHT * ev_shortfall_kwh
        reward_peak = -PEAK_IMPORT_WEIGHT * peak_excess_kwh
        reward_leftover_battery = self._leftover_battery_value(time_step)
        reward = reward_grid + reward_wear + reward_ev + reward_peak + reward_leftover_battery

        info = {
            "grid_import_kwh": grid_import_kwh,
            "grid_export_kwh": grid_export_kwh,
            "grid_cost": grid_cost,
            "battery_charge_kwh": battery_charge_kwh,
            "battery_discharge_kwh": battery_discharge_kwh,
            "ev_charge_kwh": ev_charge_kwh,
            "ev_shortfall_kwh": ev_shortfall_kwh,
            "peak_excess_kwh": peak_excess_kwh,
            "reward_grid": reward_grid,
            "reward_wear": reward_wear,
            "reward_ev": reward_ev,
            "reward_peak": reward_peak,
            "reward_leftover_battery": reward_leftover_battery,
            "battery_soc": self.battery_energy_kwh / BATTERY_CAPACITY_KWH,
            "ev_soc": self.ev_energy_kwh / EV_CAPACITY_KWH if ev_plugged_in else 0.0,
            "solar_kwh": solar_kwh,
            "load_kwh": load_kwh,
            "time_step": time_step,
        }

        self.time_step += 1
        terminated = self.time_step >= STEPS_PER_DAY

        return self._get_observation(), reward, terminated, False, info

    # -----------------------------------------------------------------------
    # Physics helpers
    # -----------------------------------------------------------------------

    def _is_ev_plugged_in(self, time_step):
        """
        Check whether the EV is plugged in at a timestep.

        Args:
            time_step: timestep index.

        Returns:
            True if ev_arrival_step <= time_step < ev_departure_step.
        """
        return self.profile.ev_arrival_step <= time_step < self.profile.ev_departure_step

    def _charge_ev(self, should_charge):
        """
        Charge the EV for one step, limited by charger power and free space.

        Args:
            should_charge: True if the EV is plugged in and the action charges it.

        Returns:
            Energy added to the EV in kWh.
        """
        if not should_charge:
            return 0.0

        free_space_kwh = EV_CAPACITY_KWH - self.ev_energy_kwh
        ev_charge_kwh = min(EV_MAX_ENERGY_PER_STEP_KWH, free_space_kwh)
        self.ev_energy_kwh += ev_charge_kwh
        return ev_charge_kwh

    def _charge_battery(self):
        """
        Charge the home battery for one step, limited by power and free space.

        Charging loses energy: drawing E kWh stores E * efficiency kWh.

        Returns:
            Energy drawn to charge the battery in kWh (before losses).
        """
        free_space_kwh = BATTERY_CAPACITY_KWH - self.battery_energy_kwh
        energy_drawn_kwh = min(BATTERY_MAX_ENERGY_PER_STEP_KWH,
                               free_space_kwh / BATTERY_CHARGE_EFFICIENCY)
        self.battery_energy_kwh += energy_drawn_kwh * BATTERY_CHARGE_EFFICIENCY
        return energy_drawn_kwh

    def _discharge_battery(self, house_demand_kwh):
        """
        Discharge the home battery to cover the house's own demand only.

        The battery never exports to the grid, so it supplies at most the
        demand that solar does not already cover.

        Args:
            house_demand_kwh: load plus EV charging minus solar, this step.

        Returns:
            Energy supplied by the battery in kWh.
        """
        energy_supplied_kwh = min(BATTERY_MAX_ENERGY_PER_STEP_KWH,
                                  self.battery_energy_kwh,
                                  max(0.0, house_demand_kwh))
        self.battery_energy_kwh -= energy_supplied_kwh
        return energy_supplied_kwh

    def _ev_shortfall_at_departure(self, time_step):
        """
        Measure how far the EV is below its target when it leaves.

        Args:
            time_step: the current timestep.

        Returns:
            kWh below target if this is the EV's last plugged-in step, else 0.
        """
        if time_step != self.profile.ev_departure_step - 1:
            return 0.0

        target_energy_kwh = self.profile.ev_target_soc * EV_CAPACITY_KWH
        return max(0.0, target_energy_kwh - self.ev_energy_kwh)

    def _leftover_battery_value(self, time_step):
        """
        Value the energy left in the battery at the end of the day.

        Without this, energy left at midnight would be worth nothing to the
        agent, so it would learn to run the battery empty. In a real house that
        energy covers the next night's demand.

        Args:
            time_step: the current timestep.

        Returns:
            Stored energy times LEFTOVER_BATTERY_VALUE_PER_KWH at the last
            timestep of the day, else 0.
        """
        if time_step != STEPS_PER_DAY - 1:
            return 0.0
        return self.battery_energy_kwh * LEFTOVER_BATTERY_VALUE_PER_KWH

    def _get_observation(self):
        """
        Build the observation for the current timestep.

        At the end of the day (time_step 48) there is no next half hour, so
        solar, load and the EV fields are reported as 0.

        Returns:
            Dict with the fields listed in the spec, section 6.
        """
        time_step = self.time_step
        battery_soc = self.battery_energy_kwh / BATTERY_CAPACITY_KWH

        if time_step >= STEPS_PER_DAY:
            return {
                "battery_soc": battery_soc,
                "ev_soc": 0.0,
                "solar_kwh": 0.0,
                "load_kwh": 0.0,
                "time_step": time_step,
                "ev_plugged_in": 0,
                "steps_to_departure": 0,
            }

        ev_plugged_in = self._is_ev_plugged_in(time_step)

        return {
            "battery_soc": battery_soc,
            "ev_soc": self.ev_energy_kwh / EV_CAPACITY_KWH if ev_plugged_in else 0.0,
            "solar_kwh": float(self.profile.solar_kwh[time_step]),
            "load_kwh": float(self.profile.load_kwh[time_step]),
            "time_step": time_step,
            "ev_plugged_in": int(ev_plugged_in),
            "steps_to_departure": self.profile.ev_departure_step - time_step if ev_plugged_in else 0,
        }


# ---------------------------------------------------------------------------
# State discretisation
# ---------------------------------------------------------------------------

def find_bin(value, bin_edges):
    """
    Find which bin a value falls into.

    Args:
        value: the number to place.
        bin_edges: sorted list of boundaries. With edges [0.2, 0.4], values
            below 0.2 are bin 0, 0.2 up to 0.4 is bin 1, 0.4 and above is bin 2.

    Returns:
        Bin index from 0 to len(bin_edges).
    """
    bin_index = 0
    for edge in bin_edges:
        if value >= edge:
            bin_index += 1
    return bin_index


def ev_steps_needed_to_reach_target(ev_soc):
    """
    Count the full-power charging steps the EV still needs to reach its target.

    Args:
        ev_soc: current EV state of charge (0 to 1).

    Returns:
        Number of steps, rounded up (0 if the target is already met).
    """
    energy_needed_kwh = max(0.0, (EV_TARGET_SOC - ev_soc) * EV_CAPACITY_KWH)
    return math.ceil(energy_needed_kwh / EV_MAX_ENERGY_PER_STEP_KWH)


def find_ev_status(observation):
    """
    Summarise the EV's situation as one value based on its charging slack.

    Charging slack = steps until departure minus charging steps still needed.
    It tells the agent how many steps it can afford to skip charging and
    still meet the target. 0 or less means it must charge every remaining step
    (or can no longer meet the target).

    Args:
        observation: observation dict from the environment.

    Returns:
        One of EV_STATUS_AWAY, EV_STATUS_TARGET_MET, EV_STATUS_RELAXED,
        EV_STATUS_MODERATE, EV_STATUS_TIGHT or EV_STATUS_CRITICAL.
    """
    if observation["ev_plugged_in"] == 0:
        return EV_STATUS_AWAY

    steps_needed = ev_steps_needed_to_reach_target(observation["ev_soc"])
    if steps_needed == 0:
        return EV_STATUS_TARGET_MET

    charging_slack = observation["steps_to_departure"] - steps_needed
    if charging_slack >= EV_RELAXED_MIN_SLACK:
        return EV_STATUS_RELAXED
    if charging_slack >= EV_MODERATE_MIN_SLACK:
        return EV_STATUS_MODERATE
    if charging_slack >= EV_TIGHT_MIN_SLACK:
        return EV_STATUS_TIGHT
    return EV_STATUS_CRITICAL


def discretise_observation(observation):
    """
    Turn a real-valued observation into the state key used by the Q-table.

    The EV is described only by its status (see find_ev_status), which
    combines its charge level and time to departure into the one thing that
    matters for decisions: how urgently it needs charging.

    Args:
        observation: observation dict from the environment.

    Returns:
        Tuple (battery_bin, solar_bin, load_bin, time_step, ev_status).
    """
    battery_bin = find_bin(observation["battery_soc"], BATTERY_SOC_BIN_EDGES)
    solar_bin = find_bin(observation["solar_kwh"], SOLAR_BIN_EDGES_KWH)
    load_bin = find_bin(observation["load_kwh"], LOAD_BIN_EDGES_KWH)
    ev_status = find_ev_status(observation)

    return (battery_bin, solar_bin, load_bin, observation["time_step"], ev_status)


def find_action_id(battery_command, charge_ev):
    """
    Look up the action ID for a battery command and EV choice.

    Args:
        battery_command: BATTERY_IDLE, BATTERY_CHARGE or BATTERY_DISCHARGE.
        charge_ev: True to charge the EV.

    Returns:
        Action ID from 0 to 5.
    """
    for action_id, action in ACTION_TABLE.items():
        if action == (battery_command, charge_ev):
            return action_id
    raise ValueError(f"No action for {battery_command}, {charge_ev}")
