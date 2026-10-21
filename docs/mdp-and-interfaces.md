# MDP Specification and Shared Interfaces

**Status:** v6, 21 October 2026 (v2: EV status based on charging slack, section 6; `scenarios.py` moved before `environment.py`, section 11. v3: battery SOC in 10 bins and 400,000 training episodes, section 6. v4: new test seed range and development days, section 7; baseline charges the battery only on a worthwhile surplus, section 10. v5: value of energy left in the battery at the end of the day, section 5. v6: the agent sees the usual departure time, not the actual one, sections 4, 6 and 10; leftover value capped at the next night's need, section 5). Once the team agrees, this is frozen. Any later change must be announced to the whole team, because the environment, agents, baseline and analysis all depend on it.

**Purpose:** a single reference for how the household environment works and how every component connects, so that the three agents, the baseline, the scenario generator and the analysis can be written in parallel and still fit together.

**Owners:** environment, baseline and interfaces: Person 1. Shared agent components: Person 2. Day profiles, uncertainty levels and metrics: Person 3. Each member owns one agent: Q-learning (Person 1), SARSA and Expected SARSA (Persons 2 and 3, to confirm).

All numeric values below are proposed defaults. They live in one configuration file (`src/config.py`) so they can be changed in one place, and each one needs a justification in the report.

---

## 1. Episode

- One episode is one calendar day, from 00:00 to 24:00.
- The day is divided into 48 timesteps of 30 minutes. Timestep `t` covers the half hour starting at `t × 30` minutes, so `t = 0` is 00:00 to 00:30 and `t = 24` is 12:00 to 12:30.
- The step length is $\Delta t = 0.5$ hours. All energy values are in kWh per step.
- The episode ends after timestep 47 (`terminated = True`). This is a genuine terminal state, so the TD target at the last step is just the reward.
- Every episode uses a new **day profile** (section 7), which sets the solar, load and EV schedule for that day and the battery's starting level.

---

## 2. Physical constants

| Constant | Value | Notes |
|---|---|---|
| Battery capacity | 10 kWh | Typical home battery |
| Battery power limit | 2.5 kW, so 1.25 kWh per step | Same limit for charging and discharging |
| Battery charging efficiency $\eta$ | 0.95 | Loss applied when charging only |
| Solar system size | 6 kW | Nominal peak output after system losses: 6 × 0.85 = 5.1 kW |
| EV battery capacity | 60 kWh | |
| EV charger power | 7 kW, so 3.5 kWh per step | Typical home wall charger. No charging loss modelled |
| EV target state of charge | 80% (48 kWh) | Must be reached by departure |
| Import price | $0.30 per kWh, flat | Stored as 48 per-step values so a time-of-use tariff is a configuration change |
| Export price | $0.05 per kWh, flat | Stored as 48 per-step values |
| Peak import threshold | 4.0 kWh per step (8 kW) | Import above this is penalised |
| $\lambda_1$ (battery wear weight) | $0.02 per kWh of battery throughput | |
| $\lambda_2$ (EV shortfall weight) | $1.00 per kWh below target at departure | |
| $\lambda_3$ (peak weight) | $0.50 per kWh above the peak threshold | |
| Leftover battery value $v$ | $0.28 per kWh left in the battery at the end of the day | Import price minus the wear of using it later |

"State of charge" (SOC) means how full a battery is, as a fraction from 0 to 1.

---

## 3. Actions

Six discrete actions (`gymnasium.spaces.Discrete(6)`):

| Action ID | Battery command | EV command |
|---|---|---|
| 0 | Idle | Do not charge |
| 1 | Charge | Do not charge |
| 2 | Discharge | Do not charge |
| 3 | Idle | Charge |
| 4 | Charge | Charge |
| 5 | Discharge | Charge |

Rules applied by the environment (not by the agent):

- **Clipping.** If an action is physically impossible it does as much as it can. Charging a full battery charges nothing. Charging an EV that is not plugged in charges nothing. No extra penalty is given for impossible actions.
- **The battery may charge from the grid** (if there is no solar surplus, the energy is imported).
- **The battery may only discharge to cover the house's own demand** (household load plus EV charging that step). It never exports to the grid. Solar can still be exported.

---

## 4. Transitions

At each timestep $t$, given the action, the environment computes the following in this order.

**Inputs for this step** (from the day profile): solar energy $S_t$, household load $L_t$, and whether the EV is plugged in.

**EV charging.** If the EV is plugged in and the action charges it:

$$E^\text{EV}_\text{charge} = \min\big(P_\text{EV}\,\Delta t,\ E^\text{EV}_\text{max} - E^\text{EV}_t\big)$$

otherwise 0. Then $E^\text{EV}_{t+1} = E^\text{EV}_t + E^\text{EV}_\text{charge}$.

**Battery charging.** If the action charges the battery, the energy drawn is

$$E^\text{bat}_\text{charge} = \min\big(P_\text{bat}\,\Delta t,\ (E^\text{bat}_\text{max} - E^\text{bat}_t)/\eta\big)$$

and the battery gains $\eta\,E^\text{bat}_\text{charge}$.

**Battery discharging.** If the action discharges the battery, it covers at most the house's remaining demand:

$$E^\text{bat}_\text{discharge} = \min\big(P_\text{bat}\,\Delta t,\ E^\text{bat}_t,\ \max(0,\ L_t + E^\text{EV}_\text{charge} - S_t)\big)$$

**Battery update.**

$$E^\text{bat}_{t+1} = E^\text{bat}_t + \eta\,E^\text{bat}_\text{charge} - E^\text{bat}_\text{discharge}$$

**Grid balance.**

$$\text{net}_t = L_t + E^\text{EV}_\text{charge} + E^\text{bat}_\text{charge} - S_t - E^\text{bat}_\text{discharge}$$

$$\text{import}_t = \max(0, \text{net}_t), \qquad \text{export}_t = \max(0, -\text{net}_t)$$

**Where the randomness comes from.** The battery, EV and grid equations above are deterministic. All randomness in $p(s' \mid s, a)$ comes from the day profile: solar, load, EV arrival and departure times, EV arrival SOC, and the battery's starting SOC (section 7). None of these depend on the agent's actions.

**Knowledge assumptions.** At the start of step $t$ the agent observes the actual solar and load for step $t$, but nothing about later steps. For the EV it knows the charge level and the household's **usual** departure time (17:00), as set on the charger, but not the actual departure, which varies from day to day (section 7). On days the car leaves earlier than usual, the controller only finds out when it is gone. These are simplifying assumptions to state in the report.

---

## 5. Reward

$$R_t = -\big(\text{import}_t \cdot p^\text{import}_t - \text{export}_t \cdot p^\text{export}_t\big) - \lambda_1 \big(E^\text{bat}_\text{charge} + E^\text{bat}_\text{discharge}\big) - \lambda_2\, \text{shortfall}_t - \lambda_3 \max\big(0,\ \text{import}_t - \text{threshold}\big) + \text{leftover}_t$$

where $\text{leftover}_t = v \cdot \min(E^\text{bat}_{48},\ E_\text{night})$ at the last timestep ($t = 47$) and 0 at every other step. $E_\text{night}$ = 3.4 kWh is the nominal household load from 00:00 to 07:00, before solar covers demand again.

| Term | Meaning |
|---|---|
| Grid term | Money paid for imports minus money earned from exports this step |
| Wear term | Penalises moving energy in or out of the battery |
| EV shortfall | kWh below the EV target. Non-zero only at the EV's last plugged-in step (`departure_step - 1`), computed after that step's charging |
| Peak term | Penalises importing more than 4.0 kWh in one half hour |
| Leftover battery | Rewards energy still stored at the end of the day, at the last step only |

Each part is also returned separately in `info` (section 8) so the analysis can show what drove the reward.

**Why the leftover battery term.** The episode ends at midnight, but the house does not: energy left in the battery covers the next night's demand. Without this term the end of the day is worth nothing, and Q-learning learned to run the battery empty by midnight (0.05 kWh left on average, against 1.5 kWh for the baseline). About 1.5 kWh, worth roughly $0.45 a day, was used "for free", which was almost the whole of its apparent advantage over the baseline. The value $v$ is the import price ($0.30) minus the wear of discharging it later ($0.02), so keeping a kWh until tomorrow is worth the same as using it tonight. With this term, on 100 validation days at medium uncertainty, the baseline scores -2.57 and Q-learning -2.74 ± 0.14 (3 seeds).

**Why the cap at the next night's need.** Without a cap, buying a kWh at $0.30 and holding it to midnight earned $0.28, so filling the battery from the grid in the evening was almost free. Some training runs did this and others did not, which made grid costs swing widely (for example $1.43 against $2.77 for nearly the same reward). Energy above about 3.4 kWh would be replaced by tomorrow's solar anyway, so it is given no value.

---

## 6. State

### Observation (what the environment returns)

The environment returns real physical values as a dictionary (`gymnasium.spaces.Dict`):

| Field | Range | Meaning |
|---|---|---|
| `battery_soc` | 0 to 1 | Home battery state of charge |
| `ev_soc` | 0 to 1 | EV state of charge. 0 when the EV is not plugged in |
| `solar_kwh` | 0 or more | Solar energy this step |
| `load_kwh` | 0 or more | Household load this step |
| `time_step` | 0 to 48 | Current timestep. 48 appears only in the final (terminal) observation |
| `ev_plugged_in` | 0 or 1 | Whether the EV is plugged in this step |
| `expected_steps_to_departure` | 0 to 48 | Steps until the household's **usual** departure time (17:00), at least 1 while plugged in. 0 when not plugged in. The car may actually leave earlier or later |

The baseline and the metrics use these raw values.

### Discretised state (what the tabular agents learn from)

A shared discretiser turns an observation into a tuple of bin numbers. That tuple is the key into the Q-table.

| Variable | Bins | Edges |
|---|---|---|
| Battery SOC | 10 | 0-10%, 10-20%, ..., 90-100% |
| Solar | 4: none, low, medium, high | below 0.05, below 0.8, below 1.6, 1.6 and above (kWh per step) |
| Load | 3: low, medium, high | below 0.3, below 0.6, 0.6 and above (kWh per step) |
| Time | 48 | One per timestep (the terminal value 48 is never used in a learning update) |
| EV status | 6, based on charging slack | see below |

**EV status and charging slack.** Charging slack is the number of steps the EV can still afford to skip charging:

$$\text{slack} = \text{expected\_steps\_to\_departure} - \left\lceil \frac{(\text{target SOC} - \text{EV SOC}) \times 60\ \text{kWh}}{3.5\ \text{kWh per step}} \right\rceil$$

| EV status | Condition |
|---|---|
| 0 away | EV not plugged in |
| 1 target met | EV at or above its target |
| 2 relaxed | slack of 8 or more |
| 3 moderate | slack of 4 to 7 |
| 4 tight | slack of 1 to 3 |
| 5 critical | slack of 0 or less (must charge every remaining step, or the target can no longer be met) |

State key: `(battery_bin, solar_bin, load_bin, time_step, ev_status)`.

**Why slack instead of the plan's separate EV SOC and deadline variables.** The first version used the plan's design: EV SOC in 5 bins plus a 3-level deadline (far, medium, near). With it, Q-learning met the EV target on only about 55% of test days (medium uncertainty, test reward about -6.3 against the baseline's -2.95), because 20% SOC bins and 3 deadline bins together cannot tell the agent whether the target is still reachable. Slack combines both into exactly that information. With slack, Q-learning met the target on about 99% of test days (test reward about -3.4 after 200,000 episodes). This is a documented design iteration for the report.

**Why the usual departure time instead of the actual one.** Slack first used each day's actual departure, which assumes the controller knows exactly when the car will leave. In practice a charger knows the household's usual time, and the car sometimes leaves earlier. Since v6 the agent and the baseline both see only the usual departure (17:00); the car still leaves at its actual time, and any shortfall is penalised then. Results on 100 validation days (3 seeds for Q-learning):

| Uncertainty | Baseline reward | Baseline EV met (on early-leave days) | Q-learning reward | Q-learning EV met (on early-leave days) |
|---|---|---|---|---|
| Low | -1.66 | 100% (100%) | -1.16 ± 0.04 | 100% (100%) |
| Medium | -2.63 | 83% (66%) | -2.51 ± 0.13 | 99% (97%) |
| High | -5.91 | 64% (31%) | -5.65 ± 0.06 | 91% (83%) |

The baseline's fixed one-step margin fails when the car leaves early; Q-learning learns a larger safety margin from the shortfall penalties alone.

**Why 10 battery bins instead of 5.** With 5 bins (20% wide), one charging step stores about 1.19 kWh, only 12% of the battery, so about half the time charging does not change the agent's state and it cannot see what charging achieved. Q-learning then barely used the battery (about 3.7 kWh charged per day against the baseline's 14.0) and lost on grid cost. Results on 100 held-out validation days (medium uncertainty, default hyperparameters, 3 seeds; baseline reward -3.21, cost $2.57):

| Battery bins | Training episodes | Reward | Grid cost | EV target met |
|---|---|---|---|---|
| 5 | 200,000 | -3.92 | $3.13 | 96% |
| 5 | 400,000 | -3.94 | $3.40 | 98% |
| 10 | 200,000 | -3.34 | $2.33 | 93% |
| 20 | 200,000 | -3.85 | $2.29 | 90% |
| 10 | 400,000 | -2.78 | $2.19 | 98.7% |

Ten bins let the agent see the effect of each charging step; twenty bins create so many states that each is visited too rarely and EV learning suffers. The larger state space needs more training (400,000 episodes), while longer training alone does not help 5 bins. This is the second documented design iteration for the report.

**Size.** The full product is 10 × 4 × 3 × 48 × 6 = 34,560 states, but many combinations cannot happen (for example, high solar at night). Q-learning visits about 4,400 distinct states in 400,000 training episodes. The Q-table is stored as a dictionary, so only visited states use memory.

---

## 7. Day profiles and uncertainty

### Day profile format

One day profile fully describes the random parts of one episode:

| Field | Type | Meaning |
|---|---|---|
| `solar_kwh` | 48 floats | Solar energy for each step |
| `load_kwh` | 48 floats | Household load for each step |
| `ev_arrival_step` | int | First step the EV is plugged in |
| `ev_departure_step` | int | First step the EV is gone. It is plugged in for steps `arrival` to `departure - 1` |
| `ev_arrival_soc` | float, 0 to 1 | EV SOC when it arrives |
| `ev_target_soc` | float, 0 to 1 | EV SOC required at departure (0.8) |
| `initial_battery_soc` | float, 0 to 1 | Home battery SOC at 00:00 |

Constraint: `0 <= ev_arrival_step < ev_departure_step <= 48`.

The EV schedule models a household that works from home or part time: the car is plugged in during the day and is needed for an evening trip. There is one plug-in window per day, entirely inside the episode.

### Nominal (average) day

- **Solar:** a half-sine curve from 06:00 to 18:00, peaking at 5.1 kW around noon (2.55 kWh per step). About 39 kWh on a clear day.
- **Load:** 0.4 kW from 00:00 to 06:00, 1.0 kW from 06:00 to 09:00, 0.5 kW from 09:00 to 17:00, 1.6 kW from 17:00 to 21:00, 0.8 kW from 21:00 to 24:00. About 18 kWh per day.
- **EV:** arrives at 09:00 (step 18), leaves at 17:00 (step 34), target 80%.

### Uncertainty levels (placeholders, final values owned by Person 3)

Solar and load are perturbed as in the plan: $S'_t = S_t \cdot w \cdot (1 + \epsilon_t)$ and $L'_t = L_t \cdot (1 + \epsilon_t)$, where $w$ is a daily weather factor and $\epsilon_t$ is drawn from a normal distribution each step. Results are clipped at 0.

| Parameter | Low | Medium | High |
|---|---|---|---|
| Daily weather factor $w$ | uniform 0.9 to 1.0 | uniform 0.7 to 1.0 | uniform 0.4 to 1.0 |
| Solar noise standard deviation | 0.05 | 0.15 | 0.30 |
| Load noise standard deviation | 0.05 | 0.15 | 0.30 |
| EV arrival step | 18 ± 1 | 18 ± 2 | 18 ± 4 |
| EV departure step | 34 ± 1 | 34 ± 2 | 34 ± 4 |
| EV arrival SOC | uniform 0.45 to 0.55 | uniform 0.35 to 0.65 | uniform 0.20 to 0.70 |
| Initial battery SOC | uniform 0.2 to 0.8 | uniform 0.2 to 0.8 | uniform 0.2 to 0.8 |

Under high uncertainty some days cannot meet the EV target (for example, arriving late at 20% and leaving early). That is intended: it tests how controllers handle unavoidable shortfalls.

### Training and test days

- Training episodes generate a new random day each episode from the run's random number generator.
- Three fixed sets of days, each generated from its own seed range that training never uses (200 days per uncertainty level unless stated):

| Set | Seeds | Used for |
|---|---|---|
| Development days | from 1,000,000 | Quick checks while building (for example `python src/training.py`) |
| Validation days | from 2,000,000 (100 days) | Scoring hyperparameter tuning runs |
| Test days | from 3,000,000 | The final comparison only |

- Every controller (baseline and all agents) is evaluated on exactly the same days.
- Test days must never be used for tuning or design decisions. The first version of this spec used seeds from 1,000,000 as test days; those days were used to compare state designs (section 6), so they were retired to development days and a fresh test range was created.

---

## 8. Environment interface

The environment is a subclass of `gymnasium.Env`.

```
HouseholdEnergyEnv(uncertainty_level="medium")

reset(seed=None, options=None) -> (observation, info)
    Starts a new day. If options contains {"profile": day_profile}, that exact
    day is used (needed for the shared test set). Otherwise a new day is
    generated for the current uncertainty level.

step(action) -> (observation, reward, terminated, truncated, info)
    Advances 30 minutes. terminated is True after timestep 47.
    truncated is always False.

action_space       = Discrete(6)
observation_space  = Dict (fields in section 6)
```

### `info` returned by `step`

| Key | Meaning |
|---|---|
| `grid_import_kwh`, `grid_export_kwh` | Energy bought from and sold to the grid this step |
| `grid_cost` | Import cost minus export income this step ($) |
| `battery_charge_kwh`, `battery_discharge_kwh` | Actual battery energy moved (after clipping) |
| `ev_charge_kwh` | Actual energy put into the EV |
| `ev_shortfall_kwh` | kWh below target at departure (0 at other steps) |
| `peak_excess_kwh` | Import above the peak threshold |
| `reward_grid`, `reward_wear`, `reward_ev`, `reward_peak`, `reward_leftover_battery` | The five reward parts. They add up to `reward` |
| `battery_soc`, `ev_soc`, `solar_kwh`, `load_kwh`, `time_step` | Values for the step just simulated, for example-day plots |

---

## 9. Agent interface

Every agent class provides the same methods, so one training loop and one evaluation loop work for all three. The shared parts (Q-table, epsilon-greedy selection, epsilon decay) are written once in `src/agent_common.py`. Each agent only adds its own update rule.

```
Agent(n_actions, learning_rate, discount_factor,
      epsilon_start, epsilon_decay, epsilon_min, seed)

select_action(state, explore=True) -> action
    Epsilon-greedy when explore is True, greedy when False.

learn(state, action, reward, next_state, done) -> next_action
    Applies the agent's update rule for one transition and returns the
    action to take in next_state. If done is True, the TD target is just
    the reward and the returned action is ignored.

end_episode()
    Decays epsilon: epsilon = max(epsilon_min, epsilon * epsilon_decay).

q_table
    Dictionary from state key to an array of n_actions Q-values (zeros for
    unseen states).
```

`state` is any hashable value: the discretised tuple for our environment, or a plain integer for small Gymnasium test environments such as `CliffWalking-v0`. Agents never see raw observations.

`learn` returns the next action because the agents choose it at different points: SARSA must choose the next action before its update (the update uses it), while Q-learning and Expected SARSA choose it after their update. Letting each agent decide keeps the training loop identical for all three and keeps each agent faithful to its own pseudocode.

### Training loop (shared, one episode)

```
observation, info = env.reset()
state = discretise(observation)
action = agent.select_action(state)
done = False
while not done:
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    next_state = discretise(observation)
    action = agent.learn(state, action, reward, next_state, done)
    state = next_state
agent.end_episode()
```

### Evaluation

Any controller that provides `act(observation) -> action` can be evaluated. The baseline implements `act` directly on raw observations. A trained agent is evaluated by discretising the observation and calling `select_action(state, explore=False)`. Evaluation never calls `learn`, so agents do not keep learning on test days.

---

## 10. Rule-based baseline

The baseline makes two decisions each step using raw observations and the constants in section 2. Once agreed, these rules are frozen and not changed after seeing results.

**EV decision.** Charge the EV if it is plugged in, below its target, and either:

- the deadline is urgent: `expected_steps_to_departure <= steps_needed + 1`, where `steps_needed` is the number of full-power charging steps needed to reach the target, rounded up; or
- there is solar surplus (`solar_kwh > load_kwh`) and the home battery is full (SOC of at least 0.99).

**Battery decision.**

- If the solar surplus (`solar_kwh - load_kwh`) is more than 0.3 kWh and the battery is not full: charge.
- Otherwise, if the house needs energy this step (`load_kwh` plus any EV charging is more than `solar_kwh`) and the battery is not empty: discharge.
- Otherwise: idle.

The two decisions are combined into one of the six action IDs. In words: use solar first, store surplus in the battery before the EV, use the battery before the grid, and switch to the EV whenever its deadline becomes urgent.

**Why the 0.3 kWh surplus threshold.** The battery always charges at full power (1.25 kWh per step), so any part not covered by surplus solar $s$ is bought from the grid. Charging is worth it only if the value of the stored energy beats its cost:

$$\underbrace{0.95 \times 1.25 \times 0.30}_{\text{stored energy, used later}} > \underbrace{0.30\,(1.25 - s)}_{\text{grid top-up}} + \underbrace{0.05\,s}_{\text{lost export}} + \underbrace{0.02\,(1.25 + 0.95 \times 1.25)}_{\text{wear in and out}}$$

which holds when $s$ is more than about 0.27 kWh, rounded to 0.3. The first version charged on any surplus, which made the baseline buy grid energy to fill the battery whenever solar was barely above load (mostly 07:00 to 09:00). Fixing this strengthens the baseline (development check, medium uncertainty, 100 validation days: reward -3.21 to -2.99, grid cost $2.57 to $2.38, EV target still met on every day). A second candidate change, not letting the battery supply EV charging unless the deadline is urgent, was tested and rejected because it made the baseline worse (reward -3.78).

---

## 11. Code organisation

Code is developed as separate files in `src/`, then assembled into the single notebook the guideline requires. Each file becomes one notebook section, so the notebook contains the actual code with markdown explanations.

| Notebook section | File | Owner |
|---|---|---|
| Setup and configuration | `src/config.py` | Person 1 |
| Day profiles and uncertainty | `src/scenarios.py` | Person 3 (initial version by Person 1) |
| Environment and MDP | `src/environment.py` (includes the discretiser) | Person 1 |
| Rule-based baseline | `src/baseline.py` | Person 1 |
| Shared agent components | `src/agent_common.py` | Person 2 (initial version by Person 1) |
| Q-learning | `src/q_learning.py` | Person 1 |
| SARSA | `src/sarsa.py` | to confirm |
| Expected SARSA | `src/expected_sarsa.py` | to confirm |
| Training and evaluation | `src/training.py` | Person 1 |
| Metrics | `src/metrics.py` | Person 3 |
| Plots and tables | `src/plotting.py` | Person 3 |

Rules that keep the move into the notebook simple:

- A file may only import from files above it in this table.
- Import our own modules in one style only, for example `from config import BATTERY_CAPACITY_KWH`, so these lines can be deleted when the code moves into the notebook.
- No code runs on import. Demos and quick checks go under `if __name__ == "__main__":` or in separate test files.
- Each file starts with a short author line, which becomes the `**Author: Name**` markdown cell the guideline requires.
- All final results, plots and tables are produced in the notebook, because the guideline gives zero marks for results that appear in the presentation or report but not in the notebook.
- Any text that goes into the notebook or report (markdown, plot titles, captions) is plain and clear and does not use em dashes.
