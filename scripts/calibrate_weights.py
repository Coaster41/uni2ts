# scripts/calibrate_weights.py
"""
Compute total effective dataset length from the gift-eval pretrain corpus,
then derive appropriate weights for Chronos synthetic datasets.
"""

from pathlib import Path
from datasets import load_from_disk
from uni2ts.common.env import env

# ── All weights from gift_eval_pretrain_weighted.yaml ──
WEIGHT_MAP = {
    # Buildings900K
    "buildings_900k": 0.07049713172331126,
    # BuildingsBench
    "sceaux": 156.53911429030117,
    "borealis": 25.39199857374493,
    "ideal": 26.45917685053796,
    "bdg-2_panther": 40.069036647372776,
    "bdg-2_fox": 78.76137356782516,
    "bdg-2_rat": 77.24149753397461,
    "bdg-2_bear": 74.50802078734561,
    "smart": 87.55633398363929,
    "lcl": 61.224515184913,
    # CloudOps
    "azure_vm_traces_2017": 0.793621442692541,
    "borg_cluster_data_2011": 0.8826552014078424,
    "alibaba_cluster_trace_2018": 2.1667961908107465,
    # CMIP6
    **{f"cmip6_{y}": 0.014186655115280208 for y in range(1850, 2015, 5)},
    # ERA5
    **{f"era5_{y}": 0.01716585268948905 for y in range(1989, 2019)},
    # GluonTS
    "taxi_30min": 1.8616203622773724,
    "uber_tlc_daily": 0.8220619090134031,
    "uber_tlc_hourly": 19.718242630954066,
    "wiki-rolling_nips": 2.6546491601272133,
    "london_smart_meters_with_missing": 22.927608461787116,
    "wind_farms_with_missing": 375.55014453728455,
    "wind_power": 33835.223085502694,
    "solar_power": 33835.5661423233,
    "oikolab_weather": 457.6698173317555,
    "elecdemand": 80.13807329474555,
    "covid_mobility": 1.8776769670882492,
    "kaggle_web_traffic_weekly": 0.5214463673288238,
    "extended_web_traffic_with_missing": 0.8724512708896471,
    "m5": 4.15088221413791,
    "m4_yearly": 0.16910075664762209,
    "m1_yearly": 0.1353240489856994,
    "m1_quarterly": 0.22764187948158027,
    "monash_m3_yearly": 0.12991127435192062,
    "monash_m3_quarterly": 0.22388844073883546,
    "tourism_yearly": 0.12224503666516284,
    "m4_hourly": 3.905654946409425,
    "m4_daily": 10.78288428974575,
    "m4_weekly": 4.6748993189990085,
    "m4_monthly": 0.989376918871495,
    "m4_quarterly": 0.42200935083170876,
    "m1_monthly": 0.33280403653964513,
    "monash_m3_monthly": 0.45439173163627383,
    "monash_m3_other": 0.31369325979707874,
    "nn5_daily_with_missing": 3.361956841988469,
    "nn5_weekly": 0.4802795488554957,
    "tourism_monthly": 1.2559503914072827,
    "tourism_quarterly": 0.41914527015987135,
    "cif_2016_6": 0.19302663773049444,
    "cif_2016_12": 0.4574893413575907,
    "traffic_hourly": 79.4794041991723,
    "traffic_weekly": 0.43911273038216747,
    "australian_electricity_demand": 1055.3196249274629,
    "rideshare_with_missing": 1.7061359211723797,
    "saugeenday": 108.45627031345387,
    "sunspot_with_missing": 337.9978760297904,
    "temperature_rain_with_missing": 3.178993204329233,
    "vehicle_trips_with_missing": 0.4520147254997792,
    "weather": 42.04664408939033,
    "car_parts_with_missing": 0.1783895467177555,
    "fred_md": 3.275049114100332,
    "pedestrian_counts": 216.97560763837362,
    "hospital": 0.3293345477866256,
    "covid_deaths": 0.8324845513495257,
    "kdd_cup_2018_with_missing": 49.07836945864431,
    "bitcoin_with_missing": 20.816687874679623,
    "us_births": 33.276511599273626,
    # LargeST
    **{f"largest_{y}": 0.5979278516007128 for y in range(2017, 2022)},
    # LibCity
    "BEIJING_SUBWAY_30MIN": 7.190470960007991,
    "HZMETRO": 10.872614167900124,
    "LOOP_SEATTLE": 391.82785978038663,
    "LOS_LOOP": 156.76324474643377,
    "M_DENSE": 80.13807329474554,
    "PEMS03": 119.8777753943317,
    "PEMS04": 77.72295327764364,
    "PEMS07": 129.0991427323572,
    "PEMS08": 81.67496785108314,
    "PEMS_BAY": 238.43821259751687,
    "Q-TRAFFIC": 2.8032337802131853,
    "SHMETRO": 40.29316710350534,
    "SZ_TAXI": 13.612494641847192,
    # Others
    "kdd2022": 161.37389428043952,
    "godaddy": 0.18753772860071735,
    "favorita_sales": 1.1316201601311238,
    "favorita_transactions": 7.1498123738614945,
    "restaurant": 1.6362806757063764,
    "hierarchical_sales": 8.224215512782676,
    "china_air_quality": 60.07271910855663,
    "beijing_air_quality": 160.38592477208667,
    "residential_load_power": 467.01254136186304,
    "residential_pv_power": 543.1776768629394,
    "cdc_fluview_ilinet": 3.897308445779395,
    "cdc_fluview_who_nrevss": 2.5812707799492274,
    "project_tycho": 5.009345873382227,
    # ProEnFo
    "gfc12_load": 180.2832203675286,
    "gfc14_load": 80.13807329474555,
    "gfc17_load": 80.24785147734109,
    "spain": 160.38592477208664,
    "pdb": 80.13807329474555,
    "elf": 99.67858979675201,
    "bull": 80.2478514773411,
    "cockatoo": 80.24785147734109,
    "hog": 80.2478514773411,
    "covid19_energy": 145.96839012453881,
    # Subseasonal
    "subseasonal": 75.3352778061906,
    "subseasonal_precip": 51.79243173038835,
}

# ── Scan datasets ──
storage_path = env.LOTSA_V1_PATH
total_effective_len = 0
missing = []
rows = []

print(f"{'Dataset':<45} {'num_ts':>10} {'weight':>12} {'eff_len':>14}")
print("=" * 85)

for dataset_name, weight in sorted(WEIGHT_MAP.items()):
    ds_path = storage_path / dataset_name
    if not ds_path.exists():
        missing.append(dataset_name)
        continue

    try:
        ds = load_from_disk(str(ds_path))
        num_ts = len(ds)
        import math
        eff_len = math.ceil(num_ts * weight)
        total_effective_len += eff_len
        rows.append((dataset_name, num_ts, weight, eff_len))
        print(f"{dataset_name:<45} {num_ts:>10,} {weight:>12.4f} {eff_len:>14,}")
    except Exception as e:
        print(f"{dataset_name:<45} ERROR: {e}")
        missing.append(dataset_name)

print("=" * 85)
print(f"{'TOTAL':<45} {'':>10} {'':>12} {total_effective_len:>14,}")
print(f"\nMissing/errored datasets ({len(missing)}): {missing}")

# ── Compute Chronos weights ──
print("\n" + "=" * 85)
print("CHRONOS WEIGHT CALIBRATION")
print("=" * 85)

# Load Chronos datasets to get num_ts
chronos_datasets = {
    "chronos_tsmixup_10m": None,
    "chronos_kernel_synth_1m": None,
}

for name in chronos_datasets:
    ds_path = storage_path / name
    if ds_path.exists():
        ds = load_from_disk(str(ds_path))
        chronos_datasets[name] = len(ds)
        print(f"{name}: {len(ds):,} time series")
    else:
        print(f"{name}: NOT FOUND at {ds_path}")
        print(f"  → Build it first with the build script")

if all(v is not None for v in chronos_datasets.values()):
    tsmixup_ts = chronos_datasets["chronos_tsmixup_10m"]
    kernel_ts = chronos_datasets["chronos_kernel_synth_1m"]

    # Observation estimates (user-provided)
    tsmixup_obs = 20e9   # ~20B observations
    kernel_obs = 1e9     # ~1B observations
    total_synth_obs = tsmixup_obs + kernel_obs

    for target_pct in [0.05, 0.08, 0.10, 0.15, 0.20]:
        # synthetic_eff / (existing_eff + synthetic_eff) = target_pct
        target_synth_eff = target_pct * total_effective_len / (1 - target_pct)

        # Split budget proportional to observations
        tsmixup_eff = target_synth_eff * (tsmixup_obs / total_synth_obs)
        kernel_eff = target_synth_eff * (kernel_obs / total_synth_obs)

        w_tsmixup = tsmixup_eff / tsmixup_ts
        w_kernel = kernel_eff / kernel_ts

        print(f"\n  Target {target_pct*100:4.1f}% synthetic:")
        print(f"    chronos_tsmixup_10m   weight = {w_tsmixup:.6f}")
        print(f"    chronos_kernel_synth_1m weight = {w_kernel:.6f}")
        print(f"    Effective: tsmixup={int(tsmixup_eff):,}  kernel={int(kernel_eff):,}  "
              f"total_synth={int(tsmixup_eff+kernel_eff):,}")
else:
    print("\n  ⚠ Cannot compute weights — build the Chronos datasets first.")