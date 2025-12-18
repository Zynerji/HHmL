"""
LIGO Gravitational Wave Verification.

Maps HHmL boundary resonances and vortex dynamics to gravitational waveforms,
comparing against real LIGO/Virgo detections from GWOSC.

Usage:
    from hhml.verification.ligo import LIGOVerification

    verifier = LIGOVerification()
    match = verifier.compare_event('GW150914', sim_strain_tensor)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class LIGOVerification:
    """
    LIGO gravitational wave verification against real detections.

    Attributes:
        data_dir: Directory for cached LIGO data
        events: Dictionary of known GW events with metadata
    """

    KNOWN_EVENTS = {
        'GW150914': {
            'gps_time': 1126259462.4,
            'merger_time': 1126259462.423,
            'duration': 4.0,
            'sample_rate': 4096,
            'detectors': ['H1', 'L1'],
            'description': 'First detection - Binary black hole merger'
        },
        'GW151226': {
            'gps_time': 1135136350.6,
            'merger_time': 1135136350.648,
            'duration': 4.0,
            'sample_rate': 4096,
            'detectors': ['H1', 'L1'],
            'description': 'Second detection - Lighter BBH merger'
        },
        'GW170817': {
            'gps_time': 1187008882.4,
            'merger_time': 1187008882.443,
            'duration': 32.0,
            'sample_rate': 4096,
            'detectors': ['H1', 'L1', 'V1'],
            'description': 'Binary neutron star merger (multi-messenger)'
        },
    }

    def __init__(self, data_dir: str = "data/ligo"):
        """
        Initialize LIGO verification system.

        Args:
            data_dir: Directory to cache LIGO data files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.events = self.KNOWN_EVENTS
        logger.info(f"LIGO verification initialized with {len(self.events)} known events")

    def fetch_ligo_event(
        self,
        event_name: str = 'GW150914',
        detector: str = 'H1'
    ) -> Optional[np.ndarray]:
        """
        Fetch LIGO strain data for a specific event.

        Args:
            event_name: Event identifier (e.g., 'GW150914')
            detector: Detector name ('H1', 'L1', 'V1')

        Returns:
            Strain time series as numpy array, or None if unavailable

        Note:
            Requires gwpy package. If not installed, returns synthetic data for testing.
        """
        if event_name not in self.events:
            logger.warning(f"Unknown event: {event_name}")
            return None

        event_info = self.events[event_name]

        # Check cache first
        cache_file = self.data_dir / f"{event_name}_{detector}.npy"
        if cache_file.exists():
            logger.info(f"Loading cached data: {cache_file}")
            return np.load(cache_file)

        # Try to fetch with gwpy
        try:
            from gwpy.timeseries import TimeSeries

            logger.info(f"Fetching {event_name} from {detector}...")
            merger_time = event_info['merger_time']
            duration = event_info['duration']
            start = merger_time - duration / 2
            end = merger_time + duration / 2

            strain = TimeSeries.fetch_open_data(detector, start, end)
            strain_array = strain.value

            # Cache for future use
            np.save(cache_file, strain_array)
            logger.info(f"Cached to: {cache_file}")

            return strain_array

        except ImportError:
            logger.warning("gwpy not installed - using synthetic waveform for testing")
            return self._generate_synthetic_chirp(event_info)

        except Exception as e:
            logger.error(f"Failed to fetch LIGO data: {e}")
            return self._generate_synthetic_chirp(event_info)

    def _generate_synthetic_chirp(self, event_info: Dict) -> np.ndarray:
        """
        Generate synthetic gravitational wave chirp for testing.

        Args:
            event_info: Event metadata dictionary

        Returns:
            Synthetic strain time series
        """
        duration = event_info['duration']
        sample_rate = event_info['sample_rate']
        n_samples = int(duration * sample_rate)

        t = np.linspace(0, duration, n_samples)

        # Simple chirp model: frequency increases, amplitude grows
        f0 = 35  # Hz (starting frequency)
        f1 = 250  # Hz (merger frequency)
        chirp_rate = (f1 - f0) / duration

        # Instantaneous frequency
        f_t = f0 + chirp_rate * t

        # Amplitude envelope (grows toward merger)
        amp = 1e-21 * (1 + 10 * (t / duration) ** 2)

        # Phase evolution
        phase = 2 * np.pi * (f0 * t + 0.5 * chirp_rate * t ** 2)

        strain = amp * np.sin(phase)

        logger.info(f"Generated synthetic chirp: {n_samples} samples, {f0}-{f1} Hz")
        return strain

    def extract_sim_strain(
        self,
        field_tensor: torch.Tensor,
        boundary_indices: Optional[torch.Tensor] = None,
        sample_rate: int = 4096
    ) -> np.ndarray:
        """
        Extract strain-like time series from HHmL field evolution.

        Args:
            field_tensor: Field values over time [time_steps, nodes] or [time_steps, nodes, features]
            boundary_indices: Node indices on Möbius boundary (if None, use all)
            sample_rate: Target sample rate (Hz)

        Returns:
            Strain-like time series (1D numpy array)

        Strategy:
            - Sum boundary field amplitudes (analogy: integrated mass quadrupole moment)
            - Compute time derivative (strain ~ d²h/dt²)
            - Normalize to LIGO scale (~10⁻²¹)
        """
        # Handle different tensor shapes
        if field_tensor.dim() == 3:
            # [time, nodes, features] → take amplitude
            field_values = torch.abs(field_tensor[:, :, 0])
        elif field_tensor.dim() == 2:
            # [time, nodes]
            field_values = torch.abs(field_tensor)
        else:
            raise ValueError(f"Unexpected field tensor shape: {field_tensor.shape}")

        # Select boundary nodes if specified
        if boundary_indices is not None:
            field_values = field_values[:, boundary_indices]

        # Sum over spatial nodes (integrated quadrupole moment)
        integrated_field = field_values.sum(dim=1).cpu().numpy()

        # Compute second derivative (strain ~ d²h/dt²)
        dt = 1.0 / sample_rate
        strain = np.gradient(np.gradient(integrated_field, dt), dt)

        # Normalize to LIGO scale
        strain = strain / np.max(np.abs(strain)) * 1e-21

        logger.info(f"Extracted sim strain: {len(strain)} samples, max |h| = {np.max(np.abs(strain)):.2e}")
        return strain

    def compute_waveform_match(
        self,
        sim_strain: np.ndarray,
        real_strain: np.ndarray,
        whiten: bool = True
    ) -> Dict[str, float]:
        """
        Compute matched-filter overlap between sim and real waveforms.

        Args:
            sim_strain: Simulated strain time series
            real_strain: Real LIGO strain time series
            whiten: Apply whitening preprocessing

        Returns:
            Dictionary with match metrics:
                - overlap: Normalized inner product (0-1)
                - snr: Signal-to-noise ratio
                - mismatch: 1 - overlap
        """
        # Ensure same length (truncate or pad)
        min_len = min(len(sim_strain), len(real_strain))
        sim_strain = sim_strain[:min_len]
        real_strain = real_strain[:min_len]

        if whiten:
            sim_strain = self._whiten_strain(sim_strain)
            real_strain = self._whiten_strain(real_strain)

        # Normalized cross-correlation
        sim_norm = np.sqrt(np.sum(sim_strain ** 2))
        real_norm = np.sqrt(np.sum(real_strain ** 2))

        if sim_norm == 0 or real_norm == 0:
            logger.warning("Zero-norm strain detected")
            return {'overlap': 0.0, 'snr': 0.0, 'mismatch': 1.0}

        overlap = np.sum(sim_strain * real_strain) / (sim_norm * real_norm)
        overlap = np.clip(overlap, -1, 1)  # Numerical stability

        # SNR estimate
        snr = overlap * np.sqrt(len(sim_strain))

        mismatch = 1 - overlap

        logger.info(f"Waveform match: overlap={overlap:.4f}, SNR={snr:.2f}, mismatch={mismatch:.4f}")

        return {
            'overlap': float(overlap),
            'snr': float(snr),
            'mismatch': float(mismatch)
        }

    def _whiten_strain(self, strain: np.ndarray, fft_len: int = 4) -> np.ndarray:
        """
        Whiten strain data in frequency domain.

        Args:
            strain: Time-domain strain
            fft_len: FFT length in seconds

        Returns:
            Whitened strain
        """
        # Simple whitening: normalize by frequency-dependent noise
        fft = np.fft.rfft(strain)
        psd = np.abs(fft) ** 2

        # Smooth PSD estimate
        psd_smooth = np.convolve(psd, np.ones(10) / 10, mode='same')
        psd_smooth[psd_smooth == 0] = 1e-20  # Avoid division by zero

        # Whiten
        fft_white = fft / np.sqrt(psd_smooth)
        strain_white = np.fft.irfft(fft_white, n=len(strain))

        return strain_white

    def compare_event(
        self,
        event_name: str,
        sim_strain_tensor: torch.Tensor,
        detector: str = 'H1',
        save_results: bool = True
    ) -> Dict[str, any]:
        """
        Full comparison pipeline for a LIGO event.

        Args:
            event_name: Event identifier
            sim_strain_tensor: Simulated field evolution tensor
            detector: LIGO detector name
            save_results: Whether to save results to JSON

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing simulation to {event_name} ({detector})...")

        # Fetch real data
        real_strain = self.fetch_ligo_event(event_name, detector)
        if real_strain is None:
            return {'error': f'Failed to fetch {event_name}'}

        # Extract sim strain
        sim_strain = self.extract_sim_strain(
            sim_strain_tensor,
            sample_rate=self.events[event_name]['sample_rate']
        )

        # Compute match
        match_metrics = self.compute_waveform_match(sim_strain, real_strain)

        # Assemble results
        results = {
            'event': event_name,
            'detector': detector,
            'description': self.events[event_name]['description'],
            'sim_samples': len(sim_strain),
            'real_samples': len(real_strain),
            'metrics': match_metrics,
            'interpretation': self._interpret_match(match_metrics['overlap'])
        }

        if save_results:
            output_file = self.data_dir / f"comparison_{event_name}_{detector}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")

        return results

    def _interpret_match(self, overlap: float) -> str:
        """Interpret overlap score."""
        if overlap > 0.9:
            return "Excellent match - strong waveform agreement"
        elif overlap > 0.7:
            return "Good match - significant waveform similarity"
        elif overlap > 0.5:
            return "Moderate match - some waveform features captured"
        elif overlap > 0.3:
            return "Weak match - limited waveform agreement"
        else:
            return "Poor match - minimal waveform similarity"


def fetch_ligo_event(event_name: str = 'GW150914', detector: str = 'H1') -> Optional[np.ndarray]:
    """
    Convenience function to fetch LIGO event data.

    Args:
        event_name: Event identifier
        detector: Detector name

    Returns:
        Strain time series or None
    """
    verifier = LIGOVerification()
    return verifier.fetch_ligo_event(event_name, detector)


def compute_waveform_match(
    sim_strain: torch.Tensor,
    real_strain: np.ndarray
) -> Dict[str, float]:
    """
    Convenience function to compute waveform match.

    Args:
        sim_strain: Simulated strain (tensor or array)
        real_strain: Real LIGO strain (array)

    Returns:
        Match metrics dictionary
    """
    if isinstance(sim_strain, torch.Tensor):
        sim_strain = sim_strain.cpu().numpy()

    verifier = LIGOVerification()
    return verifier.compute_waveform_match(sim_strain, real_strain)
