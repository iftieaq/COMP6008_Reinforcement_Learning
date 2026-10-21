"""
Project configuration.

Every constant used by the environment, the day profile generator, the
baseline and the agents is defined here, so a value only ever needs to be
changed in one place. Values follow docs/mdp-and-interfaces.md.
"""

# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------

STEPS_PER_DAY = 48
STEP_HOURS = 0.5


# ---------------------------------------------------------------------------
# Home battery
# ---------------------------------------------------------------------------

BATTERY_CAPACITY_KWH = 10.0
BATTERY_POWER_KW = 2.5
BATTERY_MAX_ENERGY_PER_STEP_KWH = BATTERY_POWER_KW * STEP_HOURS
BATTERY_CHARGE_EFFICIENCY = 0.95


# ---------------------------------------------------------------------------
# Solar
# ---------------------------------------------------------------------------

SOLAR_SYSTEM_SIZE_KW = 6.0
SOLAR_SYSTEM_LOSS_FACTOR = 0.85
SOLAR_PEAK_OUTPUT_KW = SOLAR_SYSTEM_SIZE_KW * SOLAR_SYSTEM_LOSS_FACTOR
SUNRISE_HOUR = 6.0
SUNSET_HOUR = 18.0


# ---------------------------------------------------------------------------
# Household load: (start hour, end hour, average power in kW)
# ---------------------------------------------------------------------------

NOMINAL_LOAD_PERIODS = [
    (0, 6, 0.4),
    (6, 9, 1.0),
    (9, 17, 0.5),
    (17, 21, 1.6),
    (21, 24, 0.8),
]


# ---------------------------------------------------------------------------
# Electric vehicle (EV)
# ---------------------------------------------------------------------------

EV_CAPACITY_KWH = 60.0
EV_CHARGER_POWER_KW = 7.0
EV_MAX_ENERGY_PER_STEP_KWH = EV_CHARGER_POWER_KW * STEP_HOURS
EV_TARGET_SOC = 0.8

NOMINAL_EV_ARRIVAL_STEP = 18    # 09:00
NOMINAL_EV_DEPARTURE_STEP = 34  # 17:00


# ---------------------------------------------------------------------------
# Grid prices (one value per timestep, so a time-of-use tariff is easy to add)
# ---------------------------------------------------------------------------

IMPORT_PRICE_PER_KWH = [0.30] * STEPS_PER_DAY
EXPORT_PRICE_PER_KWH = [0.05] * STEPS_PER_DAY


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

PEAK_IMPORT_THRESHOLD_KWH = 4.0

BATTERY_WEAR_WEIGHT = 0.02   # lambda_1, $ per kWh moved in or out of the battery
EV_SHORTFALL_WEIGHT = 1.00   # lambda_2, $ per kWh below the EV target at departure
PEAK_IMPORT_WEIGHT = 0.50    # lambda_3, $ per kWh imported above the peak threshold


# ---------------------------------------------------------------------------
# Actions: action ID -> (battery command, charge the EV?)
# ---------------------------------------------------------------------------

BATTERY_IDLE = "idle"
BATTERY_CHARGE = "charge"
BATTERY_DISCHARGE = "discharge"

ACTION_TABLE = {
    0: (BATTERY_IDLE, False),
    1: (BATTERY_CHARGE, False),
    2: (BATTERY_DISCHARGE, False),
    3: (BATTERY_IDLE, True),
    4: (BATTERY_CHARGE, True),
    5: (BATTERY_DISCHARGE, True),
}

NUMBER_OF_ACTIONS = len(ACTION_TABLE)


# ---------------------------------------------------------------------------
# Uncertainty levels (placeholder values, final values owned by Person 3)
# ---------------------------------------------------------------------------

UNCERTAINTY_LEVELS = {
    "low": {
        "weather_factor_range": (0.9, 1.0),
        "solar_noise_std": 0.05,
        "load_noise_std": 0.05,
        "ev_time_jitter_steps": 1,
        "ev_arrival_soc_range": (0.45, 0.55),
        "initial_battery_soc_range": (0.2, 0.8),
    },
    "medium": {
        "weather_factor_range": (0.7, 1.0),
        "solar_noise_std": 0.15,
        "load_noise_std": 0.15,
        "ev_time_jitter_steps": 2,
        "ev_arrival_soc_range": (0.35, 0.65),
        "initial_battery_soc_range": (0.2, 0.8),
    },
    "high": {
        "weather_factor_range": (0.4, 1.0),
        "solar_noise_std": 0.30,
        "load_noise_std": 0.30,
        "ev_time_jitter_steps": 4,
        "ev_arrival_soc_range": (0.20, 0.70),
        "initial_battery_soc_range": (0.2, 0.8),
    },
}


# ---------------------------------------------------------------------------
# Test days (seeds that training never uses)
# ---------------------------------------------------------------------------

TEST_SEED_START = 1_000_000
NUMBER_OF_TEST_DAYS = 200


# ---------------------------------------------------------------------------
# State discretisation for the tabular agents
# ---------------------------------------------------------------------------

SOC_BIN_EDGES = [0.2, 0.4, 0.6, 0.8]          # 5 bins
SOLAR_BIN_EDGES_KWH = [0.05, 0.8, 1.6]        # none, low, medium, high
LOAD_BIN_EDGES_KWH = [0.3, 0.6]               # low, medium, high

# EV status is based on charging slack: steps to departure minus charging steps still needed
EV_STATUS_AWAY = 0
EV_STATUS_TARGET_MET = 1
EV_STATUS_RELAXED = 2
EV_STATUS_MODERATE = 3
EV_STATUS_TIGHT = 4
EV_STATUS_CRITICAL = 5
EV_RELAXED_MIN_SLACK = 8    # 8 or more spare steps
EV_MODERATE_MIN_SLACK = 4   # 4 to 7 spare steps
EV_TIGHT_MIN_SLACK = 1      # 1 to 3 spare steps; 0 or less is critical


# ---------------------------------------------------------------------------
# Default agent hyperparameters (starting point for tuning)
# ---------------------------------------------------------------------------

DEFAULT_AGENT_SETTINGS = {
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon_start": 1.0,
    "epsilon_decay": 0.99998,  # reaches about 0.05 after 150,000 episodes
    "epsilon_min": 0.01,
}

DEFAULT_TRAINING_EPISODES = 200_000
