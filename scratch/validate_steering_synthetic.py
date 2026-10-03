"""
scratch/validate_steering_synthetic.py
======================================
Synthetic steering vector & MUSIC recovery validation sweep.
Tests 12 synthetic cases across azimuth, elevation, and distance
without requiring hardware.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from acoustic.steering import SteeringFactory
from acoustic.music import MusicLocalizer
from config.settings import SAMPLE_RATE, FRAME_SIZE


def main():
    factory = SteeringFactory()
    factory.precompute_matrices()
    localizer = MusicLocalizer()

    fs = SAMPLE_RATE
    frame_len = FRAME_SIZE
    c = 343.0
    t = np.arange(frame_len) / fs
    tones = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]

    test_cases = [
        (103.0,   0.0, 1.5, "Level (0 deg) at Az 103 deg (Far Field)"),
        (103.0,  -5.0, 1.5, "Slight Down (-5 deg) at Az 103 deg"),
        (103.0,  -6.0, 1.5, "Current Physical Pitch (-6 deg) at Az 103 deg"),
        (103.0, -10.0, 1.5, "Down (-10 deg) at Az 103 deg"),
        (103.0, -15.0, 1.5, "Steep Down (-15 deg) at Az 103 deg"),
        (103.0,  10.0, 1.5, "Up (+10 deg) at Az 103 deg"),
        ( 90.0,   0.0, 1.5, "Broadside Level (0 deg)"),
        ( 90.0,  -6.0, 1.5, "Broadside Pitch Offset (-6 deg)"),
        (103.0,   0.0, 0.8, "Level (0 deg) Near Field (0.8m)"),
        (103.0,   0.0, 0.5, "Level (0 deg) Very Near Field (0.5m)"),
        (103.0,  -6.0, 0.8, "Pitch (-6 deg) Near Field (0.8m)"),
        (103.0,  -6.0, 0.5, "Pitch (-6 deg) Very Near Field (0.5m)"),
    ]

    header = "{:<35} | {:>7} | {:>7} | {:>8} | {:>7} | {:>7} | {:>7} | {:>7} | {:>6}".format(
        "Description", "True Az", "True El", "Dist (m)", "Rec Az", "Rec El", "Az Err", "El Err", "Status"
    )
    print(header)
    print("-" * 98)

    all_passed = True
    for true_az, true_el, r_src, desc in test_cases:
        az_rad = np.deg2rad(true_az)
        el_rad = np.deg2rad(true_el)
        x_s = r_src * np.cos(el_rad) * np.cos(az_rad)
        y_s = r_src * np.cos(el_rad) * np.sin(az_rad)
        z_s = r_src * np.sin(el_rad)
        src_pos = np.array([x_s, y_s, z_s])

        mic_pos = factory.microphone_geometry.pos_total
        dist = np.linalg.norm(src_pos[:, None] - mic_pos, axis=0)
        dist_0 = np.linalg.norm(src_pos)
        tau = (dist - dist_0) / c

        sig = np.zeros((frame_len, 16), dtype=np.float64)
        for tone in tones:
            omega = 2.0 * np.pi * tone
            for m in range(16):
                sig[:, m] += np.cos(omega * (t - tau[m]))

        localizer.reset_csm_state()
        heatmap, peak = localizer.localize_with_peak(sig)

        az_err = peak["azimuth"] - true_az
        el_err = peak["elevation"] - true_el
        passed = abs(az_err) <= 0.5 and abs(el_err) <= 0.5
        if not passed:
            all_passed = False

        status_str = "PASS" if passed else "FAIL"
        row = "{:<35} | {:>7.1f} | {:>7.1f} | {:>8.1f} | {:>7.1f} | {:>7.1f} | {:>+7.2f} | {:>+7.2f} | {:>6}".format(
            desc, true_az, true_el, r_src, peak["azimuth"], peak["elevation"], az_err, el_err, status_str
        )
        print(row)

    print("-" * 98)
    summary = "ALL 12/12 PASSED" if all_passed else "FAILED"
    print(f"Synthetic Steering / MUSIC Sweep Result: {summary}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
