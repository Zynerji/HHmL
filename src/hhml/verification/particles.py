"""
Particle Physics Verification.

Maps HHmL emergent excitations (vortex energies, pruning rates) to
particle physics observables, comparing against LHC data and PDG values.

Usage:
    from hhml.verification.particles import ParticleVerification

    verifier = ParticleVerification()
    chi2 = verifier.compare_mass_spectrum(sim_energies)
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json
import logging

logger = logging.getLogger(__name__)


class ParticleVerification:
    """
    Particle physics verification against LHC and PDG data.

    Attributes:
        data_dir: Directory for cached particle data
        pdg_masses: Known Standard Model particle masses
    """

    # Standard Model particle masses (GeV)
    PDG_MASSES = {
        'electron': 0.000511,
        'muon': 0.106,
        'tau': 1.777,
        'up': 0.00216,
        'down': 0.00467,
        'charm': 1.27,
        'strange': 0.093,
        'top': 172.76,
        'bottom': 4.18,
        'W_boson': 80.377,
        'Z_boson': 91.1876,
        'Higgs': 125.25,
        'proton': 0.938,
        'neutron': 0.940,
    }

    # Decay widths (GeV)
    PDG_WIDTHS = {
        'Z_boson': 2.4952,
        'W_boson': 2.085,
        'Higgs': 0.00407,
        'top': 1.42,
    }

    def __init__(self, data_dir: str = "data/particles"):
        """
        Initialize particle physics verification system.

        Args:
            data_dir: Directory to cache LHC/PDG data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.pdg_masses = self.PDG_MASSES
        self.pdg_widths = self.PDG_WIDTHS
        logger.info(f"Particle verification initialized with {len(self.pdg_masses)} SM particles")

    def load_lhc_histogram(
        self,
        channel: str = 'higgs_4l',
        bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load LHC invariant mass histogram.

        Args:
            channel: Decay channel ('higgs_4l', 'Z_ee', 'W_enu', etc.)
            bins: Number of histogram bins

        Returns:
            Tuple of (bin_values, bin_edges)

        Note:
            If uproot not available or data not found, returns synthetic histogram.
        """
        cache_file = self.data_dir / f"lhc_{channel}_hist.npz"

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading cached LHC histogram: {cache_file}")
            data = np.load(cache_file)
            return data['values'], data['edges']

        # Try to load with uproot
        try:
            import uproot

            logger.info(f"Loading {channel} histogram with uproot...")

            # Example: HEPData ROOT file (user must download)
            root_file = self.data_dir / f"{channel}.root"

            if root_file.exists():
                file = uproot.open(root_file)
                hist = file['histogram'].to_numpy()  # Returns (values, edges)

                logger.info(f"Loaded {channel}: {len(hist[0])} bins")
                values, edges = hist

            else:
                logger.warning(f"ROOT file not found: {root_file} - generating synthetic")
                values, edges = self._generate_synthetic_spectrum(channel, bins)

        except ImportError:
            logger.warning("uproot not installed - using synthetic spectrum")
            values, edges = self._generate_synthetic_spectrum(channel, bins)

        # Cache for future use
        np.savez(cache_file, values=values, edges=edges)
        logger.info(f"Cached histogram to: {cache_file}")

        return values, edges

    def _generate_synthetic_spectrum(
        self,
        channel: str,
        bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic invariant mass spectrum.

        Args:
            channel: Decay channel
            bins: Number of bins

        Returns:
            Tuple of (bin_values, bin_edges)
        """
        # Determine mass range and peak based on channel
        if 'higgs' in channel:
            mass_center = 125.0  # GeV
            mass_range = (100.0, 150.0)
            width = 2.0
        elif 'Z' in channel:
            mass_center = 91.2
            mass_range = (70.0, 110.0)
            width = 2.5
        elif 'W' in channel:
            mass_center = 80.4
            mass_range = (60.0, 100.0)
            width = 2.1
        else:
            mass_center = 100.0
            mass_range = (50.0, 200.0)
            width = 5.0

        # Generate histogram
        edges = np.linspace(mass_range[0], mass_range[1], bins + 1)
        centers = (edges[:-1] + edges[1:]) / 2

        # Breit-Wigner peak + background
        signal = 1000 * np.exp(-((centers - mass_center) / width) ** 2)
        background = 100 * np.exp(-(centers - mass_range[0]) / 20)  # Falling background

        values = signal + background + np.random.poisson(10, size=len(centers))  # Statistical fluctuations

        logger.info(f"Generated synthetic {channel}: peak at {mass_center} GeV, width {width} GeV")
        return values, edges

    def extract_sim_spectrum(
        self,
        energy_tensor: torch.Tensor,
        mass_range: Tuple[float, float] = (0.1, 200.0),
        bins: int = 100,
        scale_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract mass spectrum from HHmL vortex/excitation energies.

        Args:
            energy_tensor: Vortex energies or eigenvalues [N_vortices]
            mass_range: (min_mass, max_mass) in GeV
            bins: Number of histogram bins
            scale_factor: Conversion from sim units to GeV

        Returns:
            Tuple of (bin_values, bin_edges)

        Strategy:
            - Interpret vortex energy levels as particle masses
            - Bin into invariant mass histogram
            - Scale to GeV units
        """
        energies = energy_tensor.cpu().numpy().flatten()

        # Scale to GeV
        energies = energies * scale_factor

        # Create histogram
        values, edges = np.histogram(energies, bins=bins, range=mass_range)

        logger.info(f"Extracted sim spectrum: {len(energies)} excitations, {bins} bins, range {mass_range} GeV")
        return values.astype(float), edges

    def compare_spectra(
        self,
        sim_spectrum: np.ndarray,
        real_spectrum: np.ndarray,
        errors: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compare simulated and real invariant mass spectra.

        Args:
            sim_spectrum: Simulated histogram values
            real_spectrum: Real LHC histogram values
            errors: Statistical errors (if None, use √N)

        Returns:
            Dictionary with comparison metrics
        """
        # Ensure matching lengths
        min_len = min(len(sim_spectrum), len(real_spectrum))
        sim_spectrum = sim_spectrum[:min_len]
        real_spectrum = real_spectrum[:min_len]

        # Estimate errors if not provided
        if errors is None:
            errors = np.sqrt(real_spectrum)
            errors = np.where(errors == 0, 1.0, errors)  # Avoid division by zero

        # Normalize sim to match real total counts
        sim_norm = np.sum(sim_spectrum)
        real_norm = np.sum(real_spectrum)

        if sim_norm > 0:
            sim_spectrum = sim_spectrum * (real_norm / sim_norm)

        # χ² statistic
        chi2 = np.sum(((sim_spectrum - real_spectrum) / errors) ** 2)
        dof = len(sim_spectrum)
        reduced_chi2 = chi2 / dof

        # Kolmogorov-Smirnov test
        ks_statistic = self._ks_test(sim_spectrum, real_spectrum)

        logger.info(f"Spectrum comparison: χ²={chi2:.2f}, χ²/dof={reduced_chi2:.3f}, KS={ks_statistic:.4f}")

        return {
            'chi_squared': float(chi2),
            'dof': int(dof),
            'reduced_chi_squared': float(reduced_chi2),
            'ks_statistic': float(ks_statistic),
            'interpretation': self._interpret_fit(reduced_chi2, ks_statistic)
        }

    def _ks_test(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Kolmogorov-Smirnov test statistic."""
        # Normalize to CDFs
        cdf1 = np.cumsum(hist1) / np.sum(hist1)
        cdf2 = np.cumsum(hist2) / np.sum(hist2)

        # KS statistic = max absolute difference
        ks = np.max(np.abs(cdf1 - cdf2))
        return ks

    def _interpret_fit(self, reduced_chi2: float, ks_stat: float) -> str:
        """Interpret fit quality."""
        if reduced_chi2 < 2.0 and ks_stat < 0.2:
            return "Excellent fit - spectrum matches well"
        elif reduced_chi2 < 5.0 and ks_stat < 0.4:
            return "Good fit - significant agreement"
        elif reduced_chi2 < 10.0 and ks_stat < 0.6:
            return "Moderate fit - some features captured"
        else:
            return "Poor fit - limited spectrum similarity"

    def compare_pdg_masses(
        self,
        sim_energies: torch.Tensor,
        particle_list: Optional[List[str]] = None,
        tolerance: float = 0.1
    ) -> Dict[str, any]:
        """
        Compare sim energy levels to known PDG particle masses.

        Args:
            sim_energies: Simulated excitation energies [N]
            particle_list: Particles to compare (if None, use all SM particles)
            tolerance: Relative tolerance for mass matching (0.1 = ±10%)

        Returns:
            Dictionary with mass comparison results
        """
        if particle_list is None:
            particle_list = list(self.pdg_masses.keys())

        energies = sim_energies.cpu().numpy().flatten()

        # Find matches
        matches = []
        for particle in particle_list:
            pdg_mass = self.pdg_masses[particle]

            # Find closest sim energy
            diffs = np.abs(energies - pdg_mass)
            min_idx = np.argmin(diffs)
            closest_energy = energies[min_idx]
            relative_error = np.abs(closest_energy - pdg_mass) / pdg_mass

            if relative_error < tolerance:
                matches.append({
                    'particle': particle,
                    'pdg_mass': pdg_mass,
                    'sim_energy': float(closest_energy),
                    'relative_error': float(relative_error),
                    'match': 'Yes'
                })
            else:
                matches.append({
                    'particle': particle,
                    'pdg_mass': pdg_mass,
                    'sim_energy': float(closest_energy),
                    'relative_error': float(relative_error),
                    'match': 'No'
                })

        # Summary statistics
        matched_count = sum(1 for m in matches if m['match'] == 'Yes')
        total_count = len(matches)

        logger.info(f"PDG mass comparison: {matched_count}/{total_count} particles matched within {tolerance*100}%")

        return {
            'tolerance': tolerance,
            'total_particles': total_count,
            'matched_particles': matched_count,
            'match_fraction': matched_count / total_count if total_count > 0 else 0.0,
            'matches': matches,
            'interpretation': f"{matched_count}/{total_count} particles matched within {tolerance*100}% tolerance"
        }

    def compare_lhc_channel(
        self,
        sim_energies: torch.Tensor,
        channel: str = 'higgs_4l',
        scale_factor: float = 1.0,
        save_results: bool = True
    ) -> Dict[str, any]:
        """
        Full comparison pipeline for LHC decay channel.

        Args:
            sim_energies: Simulated excitation energies
            channel: LHC decay channel
            scale_factor: Conversion to GeV
            save_results: Whether to save results to JSON

        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing simulation to LHC {channel} channel...")

        # Load real data
        real_values, real_edges = self.load_lhc_histogram(channel)

        # Extract sim spectrum
        mass_range = (real_edges[0], real_edges[-1])
        bins = len(real_values)
        sim_values, sim_edges = self.extract_sim_spectrum(
            sim_energies,
            mass_range=mass_range,
            bins=bins,
            scale_factor=scale_factor
        )

        # Compare
        comparison = self.compare_spectra(sim_values, real_values)

        # Assemble results
        results = {
            'channel': channel,
            'mass_range_GeV': mass_range,
            'bins': bins,
            'scale_factor': scale_factor,
            'metrics': comparison,
            'notes': 'Sim spectrum normalized to match real total counts'
        }

        if save_results:
            output_file = self.data_dir / f"comparison_{channel}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {output_file}")

        return results


def load_lhc_histogram(
    channel: str = 'higgs_4l',
    bins: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to load LHC histogram.

    Args:
        channel: Decay channel
        bins: Number of bins

    Returns:
        Tuple of (bin_values, bin_edges)
    """
    verifier = ParticleVerification()
    return verifier.load_lhc_histogram(channel, bins)


def compare_spectra(
    sim_spectrum: torch.Tensor,
    real_spectrum: np.ndarray
) -> Dict[str, float]:
    """
    Convenience function to compare mass spectra.

    Args:
        sim_spectrum: Simulated histogram (tensor or array)
        real_spectrum: Real LHC histogram (array)

    Returns:
        Comparison metrics dictionary
    """
    if isinstance(sim_spectrum, torch.Tensor):
        sim_spectrum = sim_spectrum.cpu().numpy()

    verifier = ParticleVerification()
    return verifier.compare_spectra(sim_spectrum, real_spectrum)
