"""
Data acquisition pipeline (load cell + motor angle + tunnel conditions; no accelerometers):

    Load angles file
        -> for each angle:
             move motor to angle, wait for settle
             capture load-cell data for SAMPLE_DURATION_S
             record the ENCODER angle averaged over the capture window
             save to <output_dir>/angle_<value>deg_load.csv
        -> repeat until all angles done

Tunnel conditions (pitot-static, temperature, static pressure) are sampled
in the background for the whole sweep (wind_tunnel_conditions.TunnelConditionsCollector)
and the run-averaged values are written to run_metadata.json and
tunnel_conditions_summary.csv in the run folder.

Outputs in OUTPUT_DIR:
    angle_<+000.00>deg_load.csv   load-cell capture (header holds t_start_perf_counter)
    coordinate_frame_angles.csv   one row per angle: encoder angle, frame angles, load stats
    encoder_log.txt               continuous encoder log (header holds start_time_perf_counter)
    run_metadata.json             settings, zero/tare, tunnel conditions, per-angle rows
    tunnel_conditions_summary.csv
    live_diagnostics.png

Requirements:
    pip install pyserial pandas openpyxl nidaqmx numpy matplotlib
"""

import csv
import json
import math
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from motor import Motor
from utils import coordinate_transforms as ct
import scripts.aquire_data.load_cell as lc
import wind_tunnel_conditions as wtc
from utils.live_variance_plot import LivePlotter

# --------------------------------------------------
# FOR YOU TO UPDATE BEFORE RUNS
# --------------------------------------------------

MOTOR_PORT = "COM11"
MOTOR_BAUD = 115200

LOADCELL_DEVICE = "Dev11"
LOADCELL_MODEL = "400 N, calibrated 2019"
LOADCELL_SAMPLE_RATE_HZ = 2000.0
RUN_LOAD_CELL_TARE = True          # measure a load-cell tare at the start of the run (fan off)
TARE_DURATION_S = 5.0

RUN_ZERO_MEASUREMENT = True        # fan-off tare of the pitot dP channels (needs RUN_TUNNEL_CONDITIONS)
ZERO_DURATION_S = 15
ZERO_WARN_V = 10

RUN_TUNNEL_CONDITIONS = True
AIRFOIL_CHORD_M = 0.0138           # Reynolds number length scale

BLADE_CHORD_AVG_M = None
BLADE_SPAN_M = None
BLADE_MASS_KG = None

ANGLES_FILE = "scripts/aquire_data/test_angles.xlsx"
OUTPUT_DIR = "Data/Production_10-8-26/run_name_here"

SAMPLE_DURATION_S = 60
SETTLE_TIME_S = 2

SWEEP_ANGLE_DEG = -30.0
MOUNTING_ANGLE_DEG = -5.895 + 90

SETTLE_TOLERANCE_DEG = 0.1
SETTLE_TIMEOUT_S = 30
POST_SETTLE_S = 0.2

LOAD_CLIP_WARN_FRACTION = 0.01

# --------------------------------------------------
# Helpers
# --------------------------------------------------


def load_angles(path):
    """Target angles (deg) from .xlsx, .csv or .txt (column with 'angle' in its name, else first)."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        with open(path) as f:
            return [float(line.strip()) for line in f if line.strip()]
    cols = [c for c in df.columns if "angle" in str(c).lower()]
    return df[cols[0] if cols else df.columns[0]].astype(float).tolist()


def run_zero_procedure():
    """Interactive fan-off zero of the pitot channels, then wait for fan on."""
    while True:
        input("\nTurn the tunnel fan OFF and let the air settle.\n"
              f"Press Enter to start the {ZERO_DURATION_S:.0f} s zero measurement... ")
        zero = wtc.measure_zero_offsets(ZERO_DURATION_S)
        print(f"  Freestream dP offset: {zero.freestream_dp_v:+.5f} V "
              f"(std {zero.freestream_dp_std_v:.5f}, drift {zero.freestream_dp_drift_v:+.5f})")
        print(f"  Traverse   dP offset: {zero.traverse_dp_v:+.5f} V "
              f"(std {zero.traverse_dp_std_v:.5f}, drift {zero.traverse_dp_drift_v:+.5f})")
        if max(abs(zero.freestream_dp_v), abs(zero.traverse_dp_v)) > ZERO_WARN_V:
            print(f"  WARN: an offset exceeds {ZERO_WARN_V} V -- is the fan really off?")
        if input("Press Enter to accept, or type 'r' to repeat: ").strip().lower() != "r":
            break
    return zero


def _json_safe(o):
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def write_run_metadata(path, meta):
    """Atomic write so a crash mid-write can't corrupt the file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_json_safe(meta), f, indent=2, default=str)
    os.replace(tmp, path)


def capture_load(duration_s, csv_path, tare, on_poll=None):
    """Run the load-cell capture in a worker thread; service the plot from the main thread."""
    result, errors = {}, []

    def work():
        try:
            result.update(lc.collect_load_for_duration(
                LOADCELL_DEVICE, LOADCELL_SAMPLE_RATE_HZ, duration_s, csv_path, tare=tare))
        except Exception as e:
            errors.append(e)

    t = threading.Thread(target=work)
    t.start()
    while t.is_alive():
        if on_poll is not None:
            on_poll()
        time.sleep(0.05)
    t.join()
    if errors:
        raise RuntimeError(f"Load-cell capture failed: {errors[0]}")
    return result


# --------------------------------------------------
# Main sweep
# --------------------------------------------------

ANGLE_LOG_FIELDS = [
    "index", "motor_angle_requested_deg", "motor_angle_actual_deg",
    "motor_angle_std_deg", "motor_angle_n_samples", "motor_angle_source",
    "sweep_angle_deg", "mounting_angle_deg",
    "inclination_angle_deg", "inclination_angle_deg_shifted",
    "angle_of_attack_deg", "angle_of_attack_deg_shifted",
    "load_csv_path", "load_t_start", "load_variance", "load_clip_frac",
    "mean_Fx", "mean_Fy", "mean_Fz", "mean_Mx", "mean_My", "mean_Mz",
]


def main():
    angles = load_angles(ANGLES_FILE)
    print(f"Loaded {len(angles)} target angles: {angles}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    motor = Motor(port=MOTOR_PORT, baud=MOTOR_BAUD,
                  log_file=os.path.join(OUTPUT_DIR, "encoder_log.txt"))
    bad = [a for a in angles if abs(a) > motor.max_travel_deg]
    if bad:
        raise ValueError(f"Angles outside +/-{motor.max_travel_deg:.0f} deg: {bad}")

    print(f"Frame constants: sweep={SWEEP_ANGLE_DEG:.2f} deg, mounting={MOUNTING_ANGLE_DEG:.2f} deg")

    metadata = {
        "run_dir": OUTPUT_DIR, "status": "in_progress",
        "start_time": datetime.now().isoformat(timespec="seconds"),
        "sweep_angle_deg": SWEEP_ANGLE_DEG, "mounting_angle_deg": MOUNTING_ANGLE_DEG,
        "airfoil_chord_m": AIRFOIL_CHORD_M,
        "blade_chord_avg_m": BLADE_CHORD_AVG_M, "blade_span_m": BLADE_SPAN_M,
        "blade_mass_kg": BLADE_MASS_KG,
        "sample_duration_s": SAMPLE_DURATION_S, "settle_time_s": SETTLE_TIME_S,
        "counts_per_deg": motor.counts_per_deg,
        "loadcell_model": LOADCELL_MODEL,
        "loadcell_sample_rate_hz": LOADCELL_SAMPLE_RATE_HZ,
        "loadcell_calibration_matrix": lc.CALIBRATION_MATRIX.tolist(),
        "loadcell_volts_to_matrix_input": lc.VOLTS_TO_MATRIX_INPUT,
        "loadcell_tare": None,
        "run_tunnel_conditions": RUN_TUNNEL_CONDITIONS,
        "angles_requested": angles,
        "zero_measurement": None, "tunnel_conditions": None, "angles": [],
    }
    metadata_path = os.path.join(OUTPUT_DIR, "run_metadata.json")
    write_run_metadata(metadata_path, metadata)

    plotter = None
    tunnel_collector, tunnel_started, tunnel_summary = None, False, None
    completed = False
    tare = None

    motor.start()
    angle_log_path = os.path.join(OUTPUT_DIR, "coordinate_frame_angles.csv")
    angle_log_file = open(angle_log_path, "w", newline="")
    angle_writer = csv.DictWriter(angle_log_file, fieldnames=ANGLE_LOG_FIELDS)
    angle_writer.writeheader()

    try:
        # ---- Fan-off tares (pitot + load cell), then fan on, then start tunnel collector ----
        zero = None
        if RUN_TUNNEL_CONDITIONS and RUN_ZERO_MEASUREMENT:
            zero = run_zero_procedure()      # fan is OFF during this
            metadata["zero_measurement"] = asdict(zero)
        if RUN_LOAD_CELL_TARE:
            if not (RUN_TUNNEL_CONDITIONS and RUN_ZERO_MEASUREMENT):
                input("\nFan OFF, blade mounted and still. Press Enter to tare the load cell... ")
            tare = lc.measure_tare(LOADCELL_DEVICE, LOADCELL_SAMPLE_RATE_HZ, TARE_DURATION_S)
            metadata["loadcell_tare"] = tare.tolist()
            print("  Load-cell tare: " + "  ".join(f"{l}={v:+.3f}" for l, v in zip(lc.OUTPUT_LABELS, tare)))
        write_run_metadata(metadata_path, metadata)

        if RUN_TUNNEL_CONDITIONS:
            input("\nTurn the tunnel fan ON and bring it up to speed.\n"
                  "Press Enter when ready to continue... ")
            tunnel_collector = wtc.TunnelConditionsCollector(
                length_scale_m=AIRFOIL_CHORD_M, zero_offsets=zero)
            tunnel_collector.start()
            tunnel_started = True

        plotter = LivePlotter()

        for i, angle in enumerate(angles):
            print(f"\n=== Angle {i + 1}/{len(angles)}: {angle:+.2f} deg ===")
            try:
                actual = motor.move_to_angle(
                    angle, wait=True, tolerance_deg=SETTLE_TOLERANCE_DEG,
                    timeout_s=SETTLE_TIMEOUT_S, on_poll=plotter.pump)
            except TimeoutError as e:
                print(f"  WARN: {e} -- proceeding with last known position")
                actual = motor.position()
            time.sleep(POST_SETTLE_S)

            load_csv_path = os.path.join(OUTPUT_DIR, f"angle_{angle:+07.2f}deg_load.csv")
            res = capture_load(SAMPLE_DURATION_S, load_csv_path, tare, on_poll=plotter.pump)
            if res["clip_frac"] >= LOAD_CLIP_WARN_FRACTION:
                print(f"  WARN: load-cell channel near the DAQ rail ({100 * res['clip_frac']:.1f}% of samples)")

            # ---- Motor angle from the ENCODER (never the commanded value) ----
            t0 = res["t_start"]
            enc_mean, enc_std, enc_n = motor.angle_stats(t0, t0 + SAMPLE_DURATION_S)
            if enc_n > 0:
                motor_angle, src = enc_mean, "encoder_window_mean"
            elif actual is not None:
                motor_angle, src, enc_std = actual, "encoder_settled", None
            else:
                motor_angle, src, enc_std = None, "unavailable", None
                print("  WARN: no encoder data for this angle -- frame angles and plot point skipped")

            row = {
                "index": i, "motor_angle_requested_deg": angle,
                "motor_angle_actual_deg": motor_angle, "motor_angle_std_deg": enc_std,
                "motor_angle_n_samples": enc_n, "motor_angle_source": src,
                "sweep_angle_deg": SWEEP_ANGLE_DEG, "mounting_angle_deg": MOUNTING_ANGLE_DEG,
                "load_csv_path": load_csv_path, "load_t_start": res["t_start"],
                "load_variance": res["variance"], "load_clip_frac": res["clip_frac"],
                **{f"mean_{l}": float(v) for l, v in zip(lc.OUTPUT_LABELS, res["mean"])},
            }

            if motor_angle is not None:
                fa = ct.compute_frame_angles(motor_angle, SWEEP_ANGLE_DEG, MOUNTING_ANGLE_DEG)
                for k in ("inclination_angle_deg", "inclination_angle_deg_shifted",
                          "angle_of_attack_deg", "angle_of_attack_deg_shifted"):
                    row[k] = fa[k]

            # live plot: mean load + variance vs the encoder motor angle, refreshed after every angle
            plotter.add_point(motor_angle, res["mean"], res["variance"])

            angle_writer.writerow(row)
            angle_log_file.flush()
            metadata["angles"].append(row)
            write_run_metadata(metadata_path, metadata)

        print("\nAll angles complete.")
        completed = True
        if tunnel_started:                       # stop before homing so it doesn't skew the averages
            tunnel_summary = tunnel_collector.stop()
            tunnel_started = False
    finally:
        if tunnel_started:
            try:
                tunnel_summary = tunnel_collector.stop()
            except Exception as e:
                print(f"  WARN: failed to stop tunnel collector: {e}")
        if tunnel_summary is not None:
            wtc.save_summary(tunnel_summary, output_dir=OUTPUT_DIR)
            metadata["tunnel_conditions"] = tunnel_summary

        metadata["status"] = "complete" if completed else "aborted"
        metadata["end_time"] = datetime.now().isoformat(timespec="seconds")
        write_run_metadata(metadata_path, metadata)
        angle_log_file.close()

        try:
            motor.home()
        except Exception as e:
            print(f"  WARN: failed to home motor: {e}")
        motor.stop()
        if plotter is not None:
            plotter.save(OUTPUT_DIR)
            plotter.close()


if __name__ == "__main__":
    main()
