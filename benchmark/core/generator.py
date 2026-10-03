"""
benchmark.core.generator
========================
Physics-based 16-channel synthetic audio generator for the UMA-16 array.
Uses exact spherical propagation delays and geometric spreading attenuation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple

from benchmark.models.source import SoundSource, WaveformType
from benchmark.models.signal import MultiChannelSignal
from benchmark.models.scenario import BenchmarkScenario, ReflectionPath, MicrophoneImpairment
from acoustic.steering import build_microphone_positions


class SyntheticAudioGenerator:
    """
    Synthesizes physically accurate 16-channel audio frames matching
    the exact UMA-16 geometry and acoustic coordinate convention.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        frame_size: int = 4096,
        speed_of_sound: float = 343.0,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.c = float(speed_of_sound)

        # Microphones in Acoular frame: shape (3, 16)
        # Row 0 = X, Row 1 = Y (0.0), Row 2 = Z
        self.mic_positions = build_microphone_positions()
        self.num_mics = self.mic_positions.shape[1]

    def _generate_source_waveform(
        self,
        source: SoundSource,
        num_samples: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate base 1D time-domain waveform for a single source."""
        t = np.arange(num_samples, dtype=np.float64) / float(self.sample_rate)
        waveform = np.zeros(num_samples, dtype=np.float64)

        if source.waveform == WaveformType.TONE:
            freq = source.frequency_hz
            waveform = np.sin(2.0 * np.pi * freq * t)

        elif source.waveform == WaveformType.MULTI_TONE:
            # Multi-frequency excitation across processing bands
            f_centers = [500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0]
            for f_c in f_centers:
                waveform += np.sin(2.0 * np.pi * f_c * t)

        elif source.waveform == WaveformType.CHIRP:
            f_start, f_end = source.frequency_range_hz
            k = (f_end - f_start) / (float(num_samples) / float(self.sample_rate))
            waveform = np.sin(2.0 * np.pi * (f_start * t + 0.5 * k * (t ** 2)))

        elif source.waveform == WaveformType.WHITE_NOISE:
            raw = rng.randn(num_samples)
            # Bandpass to 500-3000 Hz
            fft_r = np.fft.rfft(raw)
            freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate)
            mask = (freqs >= source.frequency_range_hz[0]) & (freqs <= source.frequency_range_hz[1])
            fft_r[~mask] = 0.0
            waveform = np.fft.irfft(fft_r, n=num_samples)

        elif source.waveform == WaveformType.PINK_NOISE:
            raw = rng.randn(num_samples)
            fft_r = np.fft.rfft(raw)
            freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate)
            freqs[0] = freqs[1]  # avoid div by zero
            fft_r = fft_r / np.sqrt(freqs)
            mask = (freqs >= 300.0) & (freqs <= 3500.0)
            fft_r[~mask] = 0.0
            waveform = np.fft.irfft(fft_r, n=num_samples)

        elif source.waveform == WaveformType.TRANSIENT_CLAP:
            # Broadband exponential decay impulse burst (representing clap/snap)
            start_idx = int(source.start_time_s * self.sample_rate)
            dur_samples = int((source.duration_s or 0.010) * self.sample_rate)
            burst = rng.randn(dur_samples)
            decay = np.exp(-np.linspace(0.0, 7.0, dur_samples))
            burst *= decay
            # Resonate across UMA-16 speech bands
            fft_b = np.fft.rfft(burst)
            b_freqs = np.fft.rfftfreq(dur_samples, 1.0 / self.sample_rate)
            b_mask = (b_freqs >= 400.0) & (b_freqs <= 3200.0)
            fft_b[~b_mask] = 0.0
            burst = np.fft.irfft(fft_b, n=dur_samples)
            end_idx = min(num_samples, start_idx + dur_samples)
            if start_idx < num_samples and end_idx > start_idx:
                waveform[start_idx:end_idx] = burst[: end_idx - start_idx]

        elif source.waveform == WaveformType.SPEECH_SYNTHETIC:
            # Multi-formant pulse train
            f_pitch = 140.0
            pulse_period = int(self.sample_rate / f_pitch)
            pulses = np.zeros(num_samples)
            pulses[::pulse_period] = 1.0
            fft_p = np.fft.rfft(pulses)
            freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate)
            h = np.zeros_like(fft_p, dtype=np.complex128)
            for f_formant, bw in [(500.0, 100.0), (1000.0, 120.0), (1500.0, 150.0), (2000.0, 180.0), (2500.0, 200.0)]:
                h += 1.0 / (1.0 + 1j * (freqs - f_formant) / (bw / 2.0))
            waveform = np.fft.irfft(fft_p * h, n=num_samples)

        elif source.waveform == WaveformType.IMPULSE:
            start_idx = int(source.start_time_s * self.sample_rate)
            if 0 <= start_idx < num_samples:
                waveform[start_idx] = 1.0

        # Normalize peak amplitude
        peak = np.max(np.abs(waveform))
        if peak > 1e-12:
            waveform = (waveform / peak) * source.amplitude

        return waveform

    def _apply_delays(
        self,
        waveform: np.ndarray,
        source: SoundSource,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply spherical wavefront propagation delays matching Acoular steering frame:
        r_m = || r_source - p_mic ||
        tau_m = (r_m - r_0) / c
        gain_m = r_0 / r_m
        """
        num_samples = len(waveform)
        # Position of source in meters (distance default 1.0 m matching Acoular grid)
        r_source = source.unit_vector * (source.distance_m if source.distance_m > 0 else 1.0)
        r0 = float(np.linalg.norm(r_source))

        # Distance from source to each microphone: shape (16,)
        # mic_positions is (3, 16)
        r_mics = np.linalg.norm(r_source[:, None] - self.mic_positions, axis=0)

        tau_seconds = (r_mics - r0) / self.c
        gains = r0 / r_mics
        tau_samples = tau_seconds * float(self.sample_rate)

        # High-precision fractional delay via FFT phase shift
        fft_w = np.fft.rfft(waveform)
        freqs = np.fft.rfftfreq(num_samples, 1.0 / self.sample_rate)

        # Phase shift: exp(-j * 2*pi * f * tau)
        phase_shifts = np.exp(-1j * 2.0 * np.pi * freqs[:, None] * tau_seconds[None, :])
        fft_channels = fft_w[:, None] * phase_shifts * gains[None, :]
        channels = np.fft.irfft(fft_channels, n=num_samples, axis=0)

        return channels, tau_samples

    def generate(
        self,
        scenario: BenchmarkScenario,
    ) -> MultiChannelSignal:
        """Generate a complete 16-channel synthesized frame for a scenario."""
        rng = np.random.RandomState(scenario.seed)
        clean_accumulator = np.zeros((scenario.frame_size, self.num_mics), dtype=np.float64)
        delays_list = []

        # 1. Primary sound sources
        for src in scenario.sources:
            src_wave = self._generate_source_waveform(src, scenario.frame_size, rng)
            src_channels, delays = self._apply_delays(src_wave, src)
            clean_accumulator += src_channels
            delays_list.append(delays)

        # 2. Multipath reflections
        for ref in scenario.reflections:
            if scenario.sources:
                ref_src = SoundSource(
                    azimuth_deg=ref.azimuth_deg,
                    elevation_deg=ref.elevation_deg,
                    distance_m=scenario.sources[0].distance_m,
                    waveform=scenario.sources[0].waveform,
                    frequency_hz=scenario.sources[0].frequency_hz,
                    amplitude=scenario.sources[0].amplitude * ref.relative_amplitude,
                    start_time_s=scenario.sources[0].start_time_s + ref.delay_ms / 1000.0,
                )
                ref_wave = self._generate_source_waveform(ref_src, scenario.frame_size, rng)
                ref_channels, _ = self._apply_delays(ref_wave, ref_src)
                clean_accumulator += ref_channels

        # 3. Microphone impairments (before noise injection)
        if scenario.impairment is not None:
            imp = scenario.impairment
            if imp.gain_mismatches is not None:
                gains = np.asarray(imp.gain_mismatches, dtype=np.float64)
                clean_accumulator *= gains[None, :]

            if imp.phase_jitter_rad is not None:
                fft_c = np.fft.rfft(clean_accumulator, axis=0)
                phases = np.exp(1j * np.asarray(imp.phase_jitter_rad)[None, :])
                clean_accumulator = np.fft.irfft(fft_c * phases, n=scenario.frame_size, axis=0)

            for dead_ch in imp.dead_channels:
                if 0 <= dead_ch < self.num_mics:
                    clean_accumulator[:, dead_ch] = 0.0

        # 4. Noise synthesis and calibrated SNR scaling
        sig_power = np.mean(clean_accumulator ** 2)
        if sig_power < 1e-15:
            sig_power = 1.0

        snr_linear = 10.0 ** (scenario.snr_db / 10.0)
        target_noise_power = sig_power / snr_linear

        raw_noise = rng.randn(scenario.frame_size, self.num_mics)
        if scenario.noise_type == "pink":
            fft_n = np.fft.rfft(raw_noise, axis=0)
            freqs = np.fft.rfftfreq(scenario.frame_size, 1.0 / self.sample_rate)
            freqs[0] = freqs[1]
            fft_n /= np.sqrt(freqs)[:, None]
            raw_noise = np.fft.irfft(fft_n, n=scenario.frame_size, axis=0)
        elif scenario.noise_type == "low_frequency":
            fft_n = np.fft.rfft(raw_noise, axis=0)
            freqs = np.fft.rfftfreq(scenario.frame_size, 1.0 / self.sample_rate)
            fft_n[freqs > 500.0, :] = 0.0
            raw_noise = np.fft.irfft(fft_n, n=scenario.frame_size, axis=0)

        # Scale noise to exact power
        noise_power = np.mean(raw_noise ** 2)
        if noise_power > 1e-15:
            scaled_noise = raw_noise * np.sqrt(target_noise_power / noise_power)
        else:
            scaled_noise = np.zeros_like(raw_noise)

        if scenario.impairment is not None:
            for dead_ch in scenario.impairment.dead_channels:
                if 0 <= dead_ch < self.num_mics:
                    scaled_noise[:, dead_ch] = 0.0

        total_signal = clean_accumulator + scaled_noise

        return MultiChannelSignal(
            data=total_signal.astype(np.float32),
            sample_rate=self.sample_rate,
            snr_db=scenario.snr_db,
            clean_data=clean_accumulator.astype(np.float32),
            noise_data=scaled_noise.astype(np.float32),
            delays_samples=delays_list[0] if delays_list else None,
        )
