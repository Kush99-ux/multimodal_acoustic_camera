"""
acoustic.music
==============
Real-time MUSIC localization for the miniDSP UMA-16.

The microphone geometry and frequency-dependent steering matrices
come from SteeringFactory. MUSIC itself only handles audio analysis,
CSM estimation, spatial spectrum calculation and peak extraction.
"""

from __future__ import annotations

import time
import numpy as np

from acoustic.audio import AudioFrame
from acoustic.models import AcousticFrame, AcousticLocalization
from acoustic.steering import SteeringFactory

from config.settings import (
    SAMPLE_RATE,
    FRAME_SIZE,
    MUSIC_SMOOTHING,
    NUM_SOURCES,
)

# ============================================================
# STAGE C — ORIGINAL-STYLE FREQUENCY BANDS
# ============================================================
ANALYSIS_BANDS = (
    (500.0, 1000.0),
    (1000.0, 1500.0),
    (1500.0, 2000.0),
    (2000.0, 2500.0),
    (2500.0, 3000.0),
)
EMA_ALPHA = 0.35
SIGMOID_GAIN = 10.0
SIGMOID_MIDPOINT = 0.5


class MusicLocalizer:
    """Frequency-domain MUSIC localizer for the UMA-16."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        smoothing: float = MUSIC_SMOOTHING,
        n_sources: int = NUM_SOURCES,
    ) -> None:

        self.sample_rate = int(sample_rate)
        self.frame_size = int(frame_size)
        self.smoothing = float(smoothing)
        self.n_sources = int(n_sources)

        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.frame_size <= 0:
            raise ValueError("frame_size must be positive.")
        if not 0.0 <= self.smoothing <= 1.0:
            raise ValueError("smoothing must be between 0 and 1.")
        if self.n_sources < 0:
            raise ValueError("n_sources cannot be negative.")

        # SteeringFactory owns the ORIGINAL UMA-16 geometry.
        self.steering_factory = SteeringFactory()
        self.steering_factory.precompute_matrices()

        self.grid_shape = tuple(self.steering_factory.grid_shape())
        self.az_values = self.steering_factory.get_azimuth_values()
        self.el_values = self.steering_factory.get_elevation_values()

        geometry = self.steering_factory.get_microphone_geometry()
        self.num_mics = int(geometry.pos_total.shape[1])

        if self.num_mics <= 0:
            raise ValueError("Microphone geometry contains no microphones.")

        self.analysis_bands = ANALYSIS_BANDS
        self.ema_alpha = EMA_ALPHA
        self._smoothed_csms: dict[int, np.ndarray] = {}
        self._band_bins = []
        
        freq_resolution = self.sample_rate / self.frame_size
        for low, high in self.analysis_bands:
            bins = np.where(
                (np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate) >= low)
                & (np.fft.rfftfreq(self.frame_size, 1.0 / self.sample_rate) < high)
            )[0]
            self._band_bins.append(bins)
            
        print(f"[MUSIC] Frequency bands: {len(self.analysis_bands)}")
        print(f"[MUSIC] Initialized with {self.num_mics} microphones")
        print(f"[MUSIC] Frame size: {self.frame_size} samples")
        print(f"[MUSIC] Sample rate: {self.sample_rate} Hz")

    # ------------------------------------------------------------------
    # Frequency / CSM
    # ------------------------------------------------------------------

    def _compute_band_csm(
        self,
        spectrum: np.ndarray,
        bins: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the spatial CSM by averaging FFT-bin outer products
        inside one frequency band.
        """
        if spectrum.ndim != 2:
            raise ValueError("spectrum must have shape (channels, frequencies).")

        if spectrum.shape[0] != self.num_mics:
            raise ValueError(
                f"Expected {self.num_mics} channels, "
                f"received {spectrum.shape[0]}."
            )

        if bins.size == 0:
            return np.zeros(
                (self.num_mics, self.num_mics),
                dtype=np.complex128,
            )

        X = spectrum[:, bins]

        csm = (X @ X.conj().T) / float(X.shape[1])

        csm = 0.5 * (csm + csm.conj().T)

        return csm

    def _smooth_csm(
        self,
        band_index: int,
        csm: np.ndarray,
    ) -> np.ndarray:

        previous = self._smoothed_csms.get(band_index)

        if previous is None:
            smoothed = csm.copy()
        else:
            smoothed = (
                self.ema_alpha * csm
                + (1.0 - self.ema_alpha) * previous
            )

        self._smoothed_csms[band_index] = smoothed

        return smoothed

    # ------------------------------------------------------------------
    # MUSIC
    # ------------------------------------------------------------------

    def _music_band_spectrum(
        self,
        csm: np.ndarray,
        steering: np.ndarray,
    ) -> np.ndarray:

        csm = csm + 1e-10 * np.eye(self.num_mics)

        eigenvalues, eigenvectors = np.linalg.eigh(csm)

        order = np.argsort(eigenvalues)[::-1]

        eigenvectors = eigenvectors[:, order]

        # Single dominant acoustic source.
        noise_subspace = eigenvectors[:, 1:]

        projection = np.einsum(
            "mf,mk->fk",
            steering.conj(),
            noise_subspace,
        )

        denominator = np.sum(
            np.abs(projection) ** 2,
            axis=1,
        )

        denominator = np.maximum(
            denominator,
            1e-12,
        )

        return 1.0 / denominator

    # ------------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------------

    def localize(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute the normalized MUSIC heatmap.

        Input shape:
            (FRAME_SIZE, 16)

        Output shape:
            (grid_y, grid_x)
        """
        window = np.hanning(self.frame_size)
        
        data = np.asarray(frame, dtype=np.float64)
        
        if data.ndim == 1:
            data = data[:, None]
            
        if data.shape[0] != self.frame_size:
            raise ValueError(
                f"Expected {self.frame_size} samples, "
                f"received {data.shape[0]}."
            )
            
        if data.shape[1] != self.num_mics:
            raise ValueError(
                f"Expected {self.num_mics} microphones, "
                f"received {data.shape[1]}."
            )
            
        windowed = data * window[:, None]
        
        fft_data = np.fft.rfft(
            windowed,
            axis=0,
        )
        
        spectrum = fft_data.T
        band_spectra = []
        
        for band_index, bins in enumerate(self._band_bins):

            csm = self._compute_band_csm(
                spectrum,
                bins,
            )

            csm = self._smooth_csm(
                band_index,
                csm,
            )

            steering = self.steering_factory.get_band_steering(
                self.analysis_bands[band_index][0],
                self.analysis_bands[band_index][1],
            )

            band_spectrum = self._music_band_spectrum(
                csm,
                steering,
            )

            band_spectra.append(band_spectrum)
            
        if not band_spectra:
            raise RuntimeError("No valid MUSIC frequency bands.")
            
        spectrum = np.mean(
            np.stack(band_spectra, axis=0),
            axis=0,
        )
        
        spectrum -= np.min(spectrum)
        peak_value = np.max(spectrum)
        if peak_value > 0:
            spectrum /= peak_value
            
        # Sigmoid contrast enhancement.
        spectrum = 1.0 / (
            1.0
            + np.exp(
                -SIGMOID_GAIN
                * (spectrum - SIGMOID_MIDPOINT)
            )
        )
        
        spectrum -= np.min(spectrum)
        peak_value = np.max(spectrum)
        if peak_value > 0:
            spectrum /= peak_value

        # Shape back into 2D heatmap format.
        heatmap = spectrum.reshape(self.grid_shape)
        heatmap = np.nan_to_num(
            heatmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return heatmap

    # ------------------------------------------------------------------
    # Peak
    # ------------------------------------------------------------------

    def get_peak_position(
        self,
        heatmap: np.ndarray,
    ) -> dict[str, float | int]:
        """Return the strongest MUSIC position in angular coordinates."""

        heatmap = np.asarray(heatmap)

        if heatmap.ndim != 2:
            raise ValueError("Heatmap must be 2-dimensional.")

        if heatmap.shape != self.grid_shape:
            raise ValueError(
                f"Expected heatmap shape {self.grid_shape}, "
                f"received {heatmap.shape}."
            )

        peak_index = int(np.argmax(heatmap))
        row, column = np.unravel_index(
            peak_index,
            heatmap.shape,
        )

        value = float(heatmap[row, column])
        
        # Pull true angles from the steering factory's domain
        azimuth = float(self.az_values[column])
        elevation = float(self.el_values[row])

        return {
            "azimuth": azimuth,
            "elevation": elevation,
            "value": value,
            "row": int(row),
            "column": int(column),
        }

    def localize_with_peak(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, float | int]]:
        """Run MUSIC and return both heatmap and strongest peak."""

        heatmap = self.localize(frame)
        return heatmap, self.get_peak_position(heatmap)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def reset_csm_state(self) -> None:
        """Clear all per-frequency temporal CSM state."""
        self._smoothed_csms.clear()

    # ------------------------------------------------------------------
    # AudioFrame integration
    # ------------------------------------------------------------------

    def localize_audio_frame(
        self,
        audio_frame: AudioFrame,
    ) -> AcousticFrame:
        """Convert one AudioFrame into a project AcousticFrame."""

        if not isinstance(audio_frame, AudioFrame):
            raise TypeError(
                "localize_audio_frame expects an AudioFrame."
            )

        data = np.asarray(audio_frame.data)

        if data.ndim != 2:
            raise ValueError(
                "AudioFrame data must be 2-dimensional."
            )

        if data.shape != (self.frame_size, self.num_mics):
            raise ValueError(
                f"Expected audio shape "
                f"({self.frame_size}, {self.num_mics}), "
                f"received {data.shape}."
            )

        start = time.perf_counter()

        heatmap, peak = self.localize_with_peak(data)

        processing_time = time.perf_counter() - start

        confidence = float(
            np.clip(peak["value"], 0.0, 1.0)
        )

        # We no longer pretend angles are physical x/y/z distances.
        localization = AcousticLocalization(
            x=0.0,
            y=0.0,
            z=0.0,
            response=float(peak["value"]),
            frequency=None,
            confidence=confidence,
            valid=True,
            source_count=self.n_sources,
        )

        return AcousticFrame(
            frame_id=audio_frame.frame_id,
            timestamp=audio_frame.timestamp,
            localization=localization,
            sample_rate=float(self.sample_rate),
            frame_size=int(self.frame_size),
            processing_time=processing_time,
            metadata={
                "source": "MusicLocalizer",
                "peak_row": int(peak["row"]),
                "peak_column": int(peak["column"]),
                "azimuth_deg": float(peak["azimuth"]),
                "elevation_deg": float(peak["elevation"]),
                "heatmap_shape": tuple(
                    int(v) for v in heatmap.shape
                ),
                "heatmap": heatmap,
                "num_microphones": self.num_mics,
                "num_frequencies": len(self.analysis_bands),
            },
        )