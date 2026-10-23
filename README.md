# Datasets: Active Thermal Control and LCL Current-Loop Stability in SiC and GaN Grid-Connected Inverters

This repository holds the raw datasets behind the study *Thermal Protection That Destabilizes the Current Loop: Sampled-Data Stability Limits, Thermal Headroom and a Physics-Informed Digital Twin for SiC and GaN Grid-Connected Inverters*.

The study looks at a 20 kW, 800 V photovoltaic inverter connected to a 400 V, 50 Hz grid through an LCL filter. When active thermal control lowers the switching frequency to protect the semiconductors, the sampled-data current loop can lose stability. The datasets cover the switched-model telemetry used to train and test the digital twin, independent circuit-level telemetry, the Monte Carlo study of SiC and GaN designs, a random-design study of the stability limit, and finite-element heatsink temperature fields.

Only data are published here. Code, processed results and figures are not part of this repository.

## Contents

| Folder | What it contains | Size |
|---|---|---|
| `data/01_switched_telemetry` | 4 800 telemetry records from the exact switched three-phase model, in four splits | 162 MB |
| `data/02_circuit_level_telemetry` | 23 telemetry records and 2 thermal-control step responses from switched circuit-level models | 4 MB |
| `data/03_monte_carlo` | 21 inverter configurations, 20 000 Monte Carlo samples each | 39 MB |
| `data/04_random_lcl_designs` | 100 000 random LCL filter and controller designs with their stability limits | 10 MB |
| `data/05_heatsink_fem_fields` | Temperature fields on the heatsink base for three designs | 5 MB |

All files are plain CSV (some gzip-compressed) or NumPy `.npz` archives that load without `allow_pickle`.

## Common symbols

| Symbol | Meaning | Unit |
|---|---|---|
| `fs0`, `fsw0` | nominal switching (and sampling) frequency | Hz (or kHz where the column name says `_kHz`) |
| `k` | frequency ratio set by thermal control, k = fsw / fsw0 | 1 |
| `kcrit` | critical frequency ratio: below it the current loop is unstable | 1 |
| `L1`, `C`, `L2` | converter-side inductance, filter capacitance, grid-side inductance | H, F, H |
| `Lg`, `L2t` | grid inductance, total grid-side inductance L2t = L2 + Lg | H |
| `Kp`, `Hi`, `Kr` | proportional gain, capacitor-current active-damping gain, resonant gain of the current controller | Ω, Ω, Ω/s |
| `SCR` | short-circuit ratio of the grid (`inf` for a stiff grid) | 1 |
| `fc_ratio`, `zeta_a` | crossover frequency over fs0, and the active-damping ratio used for tuning | 1 |
| `td` | dead time | s |
| `alpha`, `beta` | stationary-frame components | |

## 01. Switched-model telemetry (`data/01_switched_telemetry`)

Each record is one 40 ms telemetry burst logged at every controller sample: window A (20 ms, no injection) followed by window B (20 ms with a pseudo-random binary injection of 3 % of the rated peak current on the current references). The records come from an exact switched model with the true filter and grid parameters stored alongside.

| Split | Records | Content |
|---|---|---|
| `ID` | 3 000 | two-level, 40 kHz; L1, L2 within ±10 %; C from −10 to +5 %; SCR of 5 or more; resistance up to 0.5 % pu; random tuning with fc_ratio 0.025 to 0.05 and zeta_a 0.3 to 0.7 |
| `OOD_A` | 600 | wider parameters: L within ±20 %, C from −20 to +10 %, SCR 2 to 5, resistance up to 1.5 % pu |
| `OOD_B` | 600 | ID parameters plus effects the twin does not model: dead time 100 to 500 ns, actuator gain ±3 %, current-sensor gain errors ±1 % |
| `OOD_C` | 600 | other hardware: 20 kHz and 60 kHz designs built with the same design rule |

The `ID` split is stored in four parts of 750 records so that every file stays below the GitHub file-size limit.

**Telemetry archives** (`*_telemetry*.npz`):

| Array | Shape | Meaning |
|---|---|---|
| `record` | (N,) | record index, matching the `record` column of the parameter file |
| `n_window` | (N,) | samples per window (800 at 40 kHz, 400 at 20 kHz, 1 200 at 60 kHz) |
| `log` | (N, 2·W, 7) | rows 0 to n_window−1 are window A, rows n_window to 2·n_window−1 are window B; columns: time (s), i2 alpha (A), i2 beta (A), ic alpha (A), ic beta (A), cos θ, sin θ of the grid angle |
| `injection` | (N, 2, W) | alpha and beta injection on the current references during window B (A) |

W is the longest window in the split. Records with shorter windows (OOD_C at 20 kHz) are padded with `NaN`.

**Parameter files** (`*_parameters.csv`), one row per record:

| Column | Meaning |
|---|---|
| `record`, `split` | record index and split name |
| `fs0` | nominal switching frequency (Hz) |
| `fc_ratio`, `zeta_a`, `Kp`, `Hi`, `Kr` | controller tuning and the resulting gains |
| `L1n`, `Cn`, `L2n` | nominal filter values (H, F, H) |
| `dL1`, `dL2`, `dC` | relative deviation of the true filter values from nominal |
| `inv_scr` | 1 / SCR (0 for a stiff grid) |
| `L1`, `C`, `L2t` | true filter values, L2t including the grid inductance (H, F, H) |
| `r` | series resistance (Ω) |
| `td` | dead time (s) |
| `lam` | actuator (DC-link sensing) gain factor |
| `e_i2`, `e_ic` | relative gain errors of the grid-current and capacitor-current sensors |
| `fr_true` | true LCL resonance (Hz) |
| `seed`, `attempts` | random seed of the record and number of draws needed to obtain a valid design |
| `kcrit_true` | true critical frequency ratio, from the exact sampled-data model with the true effective gains |
| `diverged` | `True` if the simulation diverged (no record in this release) |

Loading example (Python):

```python
import numpy as np, pandas as pd

meta = pd.read_csv("data/01_switched_telemetry/OOD_A_parameters.csv")
tel = np.load("data/01_switched_telemetry/OOD_A_telemetry.npz")
i = 0
n = tel["n_window"][i]
window_a = tel["log"][i, :n]        # no injection
window_b = tel["log"][i, n:2 * n]   # with injection
print(meta.loc[i, "kcrit_true"], window_b.shape)
```

## 02. Circuit-level telemetry (`data/02_circuit_level_telemetry`)

These records come from switched circuit-level models built in Simscape Electrical (Specialized Power Systems) with ideal switches with on-resistance, a resistive filter, a Thevenin grid and the digital controller in a triggered subsystem. They are independent of the model used for folder 01.

`cases.csv` lists the 23 cases. Case names: `B` baseline designs, `V` variations (phase-locked loop, filter resistance, switch on-resistance, grid X/R, filter tolerance, light load, very weak grid), `D` dead time (100 ns to 1 µs), `T` three-level designs.

| Column | Meaning |
|---|---|
| `case`, `lev` | case name and number of converter levels |
| `fs0` | switching frequency (Hz) |
| `L1n`, `Cn`, `L2n` | nominal filter values used for tuning |
| `Kp`, `Hi`, `Kr` | controller gains |
| `L1`, `C`, `L2t` | filter values in the circuit model, L2t including the grid inductance |
| `R1`, `Rg` | filter and grid resistances (Ω) |
| `td` | dead time (s) |
| `kcrit_simscape` | critical frequency ratio found by bisection on the circuit model |
| `n` | number of logged samples |

`records/<case>.csv` holds the telemetry burst of each case at the nominal switching frequency (window A, then window B with injection), with columns `t_s, i2_alpha_A, i2_beta_A, ic_alpha_A, ic_beta_A, fsw_Hz, inj_alpha_A, inj_beta_A`.

`thermal_control_steps/step_to_k0.62.csv` and `step_to_k0.55.csv` are the responses of the two-level, 40 kHz baseline when thermal control steps the frequency ratio to 0.62 (stable) and to 0.55 (unstable). Columns: `t_s, i2_alpha_A, i2_beta_A, ic_alpha_A, ic_beta_A, fsw_Hz`.

## 03. Monte Carlo study (`data/03_monte_carlo`)

One file per configuration, `monte_carlo_<device>_<topology>_<fsw0>_n<parallel devices>.csv.gz`, with 20 000 samples each. Devices: Wolfspeed C3M0032120K (1200 V SiC, two-level), Wolfspeed C3M0025065K (650 V SiC, three-level) and GaN Systems GS66516T (650 V GaN, three-level, one or two in parallel).

For every sample, the heatsink is sized with nominal data, the uncertain parameters are drawn, thermal control acts on the estimated junction temperature, and the controller tuning is drawn from the robust set with the grid at its worst case.

| Column | Meaning |
|---|---|
| `dev`, `material`, `topo`, `fsw0_kHz`, `n_par` | configuration |
| `Rha_KW` | heatsink-to-ambient thermal resistance from the design rule (K/W) |
| `eta_semi_design`, `sw_share_design` | semiconductor efficiency and switching share of the losses at the design point |
| `kcrit`, `tuning_id` | critical frequency ratio of the drawn tuning, and its index in the robust set |
| `p_rsp` | spreading resistance under the device (K/W) |
| `p_rds_scale` | production spread of the on-resistance, as a factor on the typical value |
| `p_rthch` | case-to-heatsink resistance (K/W) |
| `p_h` | factor on the device share of the leg loss |
| `p_c_rip` | ripple-current coefficient |
| `p_eT` | error of the junction-temperature estimate (K) |
| `p_escale` | factor on the switching energy |
| `p_gan_exp`, `p_gan_alpha` | current exponent and temperature coefficient of the GaN switching-energy model (1 and 0 for SiC) |
| `Ta_ATC_activates` | ambient temperature at which thermal control starts to lower the frequency (°C) |
| `Ta_ATC_infeasible` | ambient temperature at which thermal control can no longer hold the junction limit (°C) |
| `Ta_unstable_floor0.25`, `Ta_unstable_floor0.5` | ambient temperature at which the current loop becomes unstable, for thermal-control floors of 0.25 and 0.5 (°C) |
| `kreq_45` to `kreq_60` | frequency ratio required by thermal control at 45, 50, 55 and 60 °C ambient |

Ambient temperatures are evaluated from 40 to 70 °C in 1 K steps. An empty cell means the event does not happen within that range.

## 04. Random LCL designs (`data/04_random_lcl_designs`)

`random_lcl_designs_100000.csv.gz` contains 100 000 random two- and three-level designs with capacitor-current active damping.

| Column | Meaning |
|---|---|
| `levels`, `fsw0` | converter levels and switching frequency (Hz) |
| `ripple`, `cap_frac`, `L2_over_L1` | filter design inputs: current ripple, capacitor size as a fraction of base, inductance ratio |
| `SCR`, `esr_pu` | grid short-circuit ratio and filter resistance (pu) |
| `fc_ratio`, `zeta_a`, `Kr_over_Kp` | controller tuning inputs |
| `L1`, `C`, `L2`, `Lg`, `L2t` | resulting filter and grid values (H, F) |
| `fr0`, `fr` | resonance on a stiff grid and with the grid inductance (Hz) |
| `Kp`, `Hi`, `Kr` | controller gains |
| `r1`, `r2` | converter-side and grid-side resistances (Ω) |
| `stable_nominal` | whether the design is stable at the nominal frequency |
| `kcrit` | critical frequency ratio from the exact discrete model |
| `n_transitions` | number of stability transitions found while sweeping k |
| `floor_6rho` | simple floor 6 fr / fs0 |
| `kcrit_closed_form_ccf` | closed-form critical ratio for capacitor-current feedback |
| `D`, `sp` | net damping coefficient and passive damping rate of the generalized floor |
| `k_sg` | generalized small-gain floor |

## 05. Heatsink temperature fields (`data/05_heatsink_fem_fields`)

`field_SiC2L40k_m4.txt`, `field_SiC3L100k_m4.txt` and `field_GaN3L100k_m4.txt` are exports of the three-dimensional heat-transfer model (COMSOL Multiphysics 6.4) at the design point, sampled on the heatsink base plane. The first lines starting with `%` are the export header. Each data row gives x (mm), y (mm) and the temperature in °C (the header label `T-273.15 (K)` is the temperature converted from kelvin to degrees Celsius).

## License

The datasets are released under the [Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt them for any purpose, including commercial use, as long as you give appropriate credit. The full license text is in `LICENSE`.

## How to cite

Please cite the associated article when it is published, and this dataset as described in `CITATION.cff`.

## Contact

Md. Sazzad Hossain
