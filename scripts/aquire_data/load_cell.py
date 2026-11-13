"""
load_cell.py

NI-DAQ load-cell acquisition for the sweep pipeline (main.py).

Load cell: 400 N, calibrated in 2019. Six strain-gauge channels AI0..AI5
(differential) -> [Fx, Fy, Fz, Mx, My, Mz] via CALIBRATION_MATRIX.

    F = CALIBRATION_MATRIX @ V  -  F0

F0 (the tare) is NOT stored here: the old ZERO_OFFSET belonged to the old
cell. Either pass a tare to collect_load_for_duration (measured with
measure_tare() at the start of a run) or leave it None (no tare; means are
then raw calibrated values, variances are unaffected).

!! Check the units your calibration expects. If the matrix was derived for
!! bridge output in mV/V (or volts after amplifier gain) rather than the raw
!! DAQ volts read here, apply that scale in VOLTS_TO_MATRIX_INPUT below.
"""

import csv
import time

import numpy as np
import nidaqmx
from nidaqmx.constants import TerminalConfiguration

N_CHANNELS = 6  # AI0 .. AI5, all DIFFERENTIAL

# # 400 N load cell - calibrated in 2019
# CALIBRATION_MATRIX = np.array([
#     [ 52.497,  0.376, -0.204, -2.305,  0.149,  0.782],
#     [  0.134, 52.278,  0.796,  0.068,  0.677,  1.018],
#     [ -0.444, -3.426, 104.655, 0.150,  0.0335, 4.513],
#     [  0.014,  0.038,  0.011, 10.495,  0.003,  0.101],
#     [ -0.022, -0.277,  0.038, -0.007, 10.586,  0.424],
#     [  0.021,  0.081,  0.044,  0.031, -0.144, 10.510],
# ])

#sn 6434 200N load cell
CALIBRATION_MATRIX = np.array([
    [ 26.504, .105, -.324, 1.074, -0.017, 0.092],
    [ 0.112, 26.665, -.132, 0.062, -0.522, -0.076],
    [ 1.617, -0.679, 53.175, -0.179, -0.484, 1.137],
    [-0.060, 0.007, -0.023, 5.350, -0.002, -0.173],
    [0.022, -0.012, -0.015, 0.003, 5.368, 0.076],
    [ 0.013, -0.026, -0.016, 0.058, 0.020, 5.358],
])

# Multiplier from DAQ volts to whatever the matrix expects (1.0 = matrix takes raw volts).
VOLTS_TO_MATRIX_INPUT = 1.0

OUTPUT_LABELS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
DAQ_RANGE_V = 10.0
CLIP_LEVEL = 0.99


def _add_channels(task, device_name):
    for ch in range(N_CHANNELS):
        task.ai_channels.add_ai_voltage_chan(
            f"{device_name}/ai{ch}",
            terminal_config=TerminalConfiguration.DIFF,
            min_val=-DAQ_RANGE_V, max_val=DAQ_RANGE_V,
        )


def calibrate_load(data, tare=None):
    """data: (n_samples, 6) volts. Returns (6, n_samples) [Fx..Mz].
    tare: optional length-6 array subtracted from the calibrated values."""
    F = CALIBRATION_MATRIX @ (data.T * VOLTS_TO_MATRIX_INPUT)
    if tare is not None:
        F = F - np.asarray(tare, dtype=float).reshape(6, 1)
    return F


def loadcell_magnitude_variance(force_xyz_rows):
    """Variance of the per-sample force magnitude |F| (rows of Fx, Fy, Fz)."""
    mags = np.linalg.norm(np.asarray(force_xyz_rows, dtype=float), axis=1)
    return float(np.var(mags))


def acquire_load(device_name, sample_rate, duration_s):
    """Finite read of all 6 channels. Returns (data (n,6) volts, t (n,))."""
    n = int(round(sample_rate * duration_s))
    with nidaqmx.Task() as task:
        _add_channels(task, device_name)
        task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=n)
        raw = task.read(number_of_samples_per_channel=n, timeout=duration_s + 10.0)
    return np.array(raw).T, np.arange(n) / sample_rate


def measure_tare(device_name, sample_rate, duration_s=5.0):
    """Mean calibrated load with the rig unloaded / fan off. Returns a length-6
    array (N, N*m) to pass as `tare`. Taring with the blade mounted also removes
    the blade's static weight from the mean forces."""
    data, _ = acquire_load(device_name, sample_rate, duration_s)
    return calibrate_load(data).mean(axis=1)


def collect_load_for_duration(device_name, sample_rate, duration_s, csv_path, tare=None):
    """Capture duration_s seconds, write csv_path, return a dict:
        n_samples, t_start (perf_counter at DAQ start), variance (Var|F|, N^2),
        mean (6,), clip_frac (largest fraction of samples near the +-10 V rail).
    CSV time column is relative to t_start; t_start_perf_counter is in the header
    (same clock as Motor's encoder log, so encoder angle can be window-averaged).
    """
    n = int(round(sample_rate * duration_s))
    with nidaqmx.Task() as task:
        _add_channels(task, device_name)
        task.timing.cfg_samp_clk_timing(sample_rate, samps_per_chan=n)
        task.start()
        t_start = time.perf_counter()
        raw = task.read(number_of_samples_per_channel=n, timeout=duration_s + 10.0)
        task.stop()

    data = np.array(raw).T            # (n, 6) volts
    F = calibrate_load(data, tare)    # (6, n)
    t = np.arange(n) / sample_rate

    with open(csv_path, "w", newline="") as f:
        f.write(f"# NI load cell (400 N), {sample_rate:.1f} Hz, t_start_perf_counter={t_start:.6f}\n")
        w = csv.writer(f)
        w.writerow(["sample", "time"] + [f"stg{i}" for i in range(N_CHANNELS)] + OUTPUT_LABELS)
        for i in range(n):
            w.writerow([i, f"{t[i]:.6f}"] + data[i].tolist() + F[:, i].tolist())

    elapsed = time.perf_counter() - t_start
    print(f"  Load cell: captured {n} samples in {elapsed:.2f}s ({n / elapsed:.0f}/s)")
    clip = float((np.abs(data) >= CLIP_LEVEL * DAQ_RANGE_V).mean(axis=0).max()) if n else 0.0
    return dict(n_samples=n, t_start=t_start,
                variance=loadcell_magnitude_variance(F[:3].T) if n else 0.0,
                mean=F.mean(axis=1) if n else np.zeros(6), clip_frac=clip)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="Dev11")
    p.add_argument("--rate", type=float, default=1000.0)
    p.add_argument("--duration", type=float, default=10.0)
    a = p.parse_args()
    data, _ = acquire_load(a.device, a.rate, a.duration)
    F = calibrate_load(data)
    for label, v in zip(OUTPUT_LABELS, F.mean(axis=1)):
        print(f"  {label}: {v: .6f}")


if __name__ == "__main__":
    main()
