"""
Automated Emergent Phenomenon Verification.

This module provides automated verification of emergent phenomena against
real-world physics data (LIGO, CMB, particles) to judge if discoveries are
truly novel and exhibit patterns similar to empirical observations.

Usage:
    from hhml.utils.emergent_verifier import EmergentVerifier

    verifier = EmergentVerifier()
    results = verifier.verify_phenomenon(
        field_tensor=final_field,
        phenomenon_type='oscillatory',
        save_results=True
    )

    if results['is_novel']:
        print("Novel emergent phenomenon detected!")
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import logging

from hhml.verification.ligo import LIGOVerification
from hhml.verification.cmb import CMBVerification
from hhml.verification.particles import ParticleVerification

logger = logging.getLogger(__name__)


class EmergentVerifier:
    """
    Automated verification system for emergent phenomena.

    Integrates LIGO, CMB, and particle physics verification to judge
    if discovered emergent patterns are novel and match real-world physics.
    """

    # Thresholds for "good" verification (strengthens novelty claim)
    THRESHOLDS = {
        'ligo': {
            'excellent': 0.9,
            'good': 0.7,
            'moderate': 0.5,
            'weak': 0.3
        },
        'cmb': {
            'excellent': 1.5,
            'good': 3.0,
            'moderate': 5.0,
            'weak': 10.0
        },
        'particles': {
            'excellent': 0.7,
            'good': 0.5,
            'moderate': 0.3,
            'weak': 0.1
        }
    }

    def __init__(self, data_dir: str = "data"):
        """
        Initialize emergent verifier.

        Args:
            data_dir: Base directory for verification data
        """
        self.data_dir = Path(data_dir)
        self.ligo = LIGOVerification(data_dir=str(self.data_dir / "ligo"))
        self.cmb = CMBVerification(data_dir=str(self.data_dir / "cmb"))
        self.particles = ParticleVerification(data_dir=str(self.data_dir / "particles"))

        logger.info("EmergentVerifier initialized")

    def verify_phenomenon(
        self,
        field_tensor: Optional[torch.Tensor] = None,
        vortex_energies: Optional[torch.Tensor] = None,
        phenomenon_type: str = 'auto',
        events: Optional[List[str]] = None,
        save_results: bool = True,
        output_dir: str = "data/verification"
    ) -> Dict[str, any]:
        """
        Automatically verify emergent phenomenon against real-world data.

        Args:
            field_tensor: Field evolution tensor [time, nodes, features] or [nodes, features]
            vortex_energies: Vortex energy levels [N_vortices] (for particle verification)
            phenomenon_type: Type of phenomenon:
                - 'oscillatory' or 'wave': LIGO comparison
                - 'spatial' or 'fluctuation': CMB comparison
                - 'energetic' or 'discrete': Particle comparison
                - 'auto': Automatic detection based on field properties
                - 'all': Run all three verifications
            events: LIGO events to compare (default: ['GW150914'])
            save_results: Whether to save results to JSON
            output_dir: Directory for output files

        Returns:
            Dictionary with verification results and novelty assessment
        """
        if field_tensor is None and vortex_energies is None:
            raise ValueError("Must provide either field_tensor or vortex_energies")

        # Auto-detect phenomenon type if requested
        if phenomenon_type == 'auto' and field_tensor is not None:
            phenomenon_type = self._detect_phenomenon_type(field_tensor)
            logger.info(f"Auto-detected phenomenon type: {phenomenon_type}")

        # Initialize results
        results = {
            'phenomenon_type': phenomenon_type,
            'verification': {},
            'is_novel': False,
            'novelty_score': 0.0,
            'interpretation': '',
            'recommendations': []
        }

        # Run appropriate verifications
        if phenomenon_type in ['oscillatory', 'wave', 'all']:
            if field_tensor is not None:
                results['verification']['ligo'] = self._verify_ligo(
                    field_tensor, events or ['GW150914']
                )

        if phenomenon_type in ['spatial', 'fluctuation', 'all']:
            if field_tensor is not None:
                results['verification']['cmb'] = self._verify_cmb(field_tensor)

        if phenomenon_type in ['energetic', 'discrete', 'all']:
            if vortex_energies is not None:
                results['verification']['particles'] = self._verify_particles(vortex_energies)
            elif field_tensor is not None:
                # Try to extract energies from field
                energies = self._extract_energies(field_tensor)
                if energies is not None:
                    results['verification']['particles'] = self._verify_particles(energies)

        # Assess novelty
        results = self._assess_novelty(results)

        # Save results
        if save_results:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            output_file = output_path / "emergent_verification.json"
            with open(output_file, 'w') as f:
                # Convert tensors to lists for JSON serialization
                json_results = self._prepare_for_json(results)
                json.dump(json_results, f, indent=2)

            logger.info(f"Verification results saved to: {output_file}")
            results['output_file'] = str(output_file)

        return results

    def _detect_phenomenon_type(self, field_tensor: torch.Tensor) -> str:
        """
        Auto-detect phenomenon type from field properties.

        Args:
            field_tensor: Field tensor

        Returns:
            Detected type: 'oscillatory', 'spatial', or 'all'
        """
        # Check for temporal dimension (oscillatory)
        has_temporal = field_tensor.dim() >= 2 and field_tensor.shape[0] > field_tensor.shape[1]

        # Check for spatial structure (non-uniform field)
        if field_tensor.dim() >= 2:
            field_values = torch.abs(field_tensor).flatten()
            spatial_variance = torch.var(field_values) / (torch.mean(field_values) + 1e-10)
            has_spatial_structure = spatial_variance > 0.1

        else:
            has_spatial_structure = False

        # Decide type
        if has_temporal and has_spatial_structure:
            return 'all'
        elif has_temporal:
            return 'oscillatory'
        elif has_spatial_structure:
            return 'spatial'
        else:
            return 'all'  # Default to all verifications

    def _verify_ligo(self, field_tensor: torch.Tensor, events: List[str]) -> Dict:
        """Run LIGO verification."""
        logger.info(f"Running LIGO verification for events: {events}")

        results = {}
        best_overlap = 0.0
        best_event = None

        for event in events:
            try:
                event_result = self.ligo.compare_event(
                    event_name=event,
                    sim_strain_tensor=field_tensor,
                    save_results=False
                )

                overlap = event_result['metrics']['overlap']
                results[event] = event_result

                if overlap > best_overlap:
                    best_overlap = overlap
                    best_event = event

            except Exception as e:
                logger.error(f"LIGO verification failed for {event}: {e}")
                results[event] = {'error': str(e)}

        # Summary
        results['best_match'] = {
            'event': best_event,
            'overlap': best_overlap,
            'quality': self._interpret_ligo(best_overlap)
        }

        return results

    def _verify_cmb(self, field_tensor: torch.Tensor) -> Dict:
        """Run CMB verification."""
        logger.info("Running CMB verification")

        try:
            results = self.cmb.compare_planck(
                sim_field_tensor=field_tensor,
                cl_type='TT',
                lmax=1000,
                save_results=False
            )

            # Add quality interpretation
            chi2_dof = results['metrics']['reduced_chi_squared']
            results['quality'] = self._interpret_cmb(chi2_dof)

            return results

        except Exception as e:
            logger.error(f"CMB verification failed: {e}")
            return {'error': str(e)}

    def _verify_particles(self, vortex_energies: torch.Tensor) -> Dict:
        """Run particle physics verification."""
        logger.info("Running particle physics verification")

        try:
            # PDG mass comparison
            pdg_results = self.particles.compare_pdg_masses(
                sim_energies=vortex_energies,
                tolerance=0.1
            )

            # LHC Higgs channel comparison
            lhc_results = self.particles.compare_lhc_channel(
                sim_energies=vortex_energies,
                channel='higgs_4l',
                scale_factor=1.0,
                save_results=False
            )

            results = {
                'pdg': pdg_results,
                'lhc': lhc_results,
                'quality': self._interpret_particles(pdg_results['match_fraction'])
            }

            return results

        except Exception as e:
            logger.error(f"Particle verification failed: {e}")
            return {'error': str(e)}

    def _extract_energies(self, field_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract energy spectrum from field tensor."""
        # Simple extraction: field magnitudes as "energies"
        energies = torch.abs(field_tensor).flatten()

        # Filter to significant values
        threshold = torch.quantile(energies, 0.5)
        energies = energies[energies > threshold]

        if len(energies) > 10:
            return energies
        else:
            return None

    def _assess_novelty(self, results: Dict) -> Dict:
        """
        Assess if phenomenon is novel based on verification results.

        A phenomenon is considered novel if it meets verification thresholds
        for at least one type AND shows good metrics.
        """
        verification = results['verification']
        scores = []

        # LIGO score
        if 'ligo' in verification and 'best_match' in verification['ligo']:
            overlap = verification['ligo']['best_match']['overlap']
            if overlap >= self.THRESHOLDS['ligo']['excellent']:
                scores.append(1.0)
            elif overlap >= self.THRESHOLDS['ligo']['good']:
                scores.append(0.7)
            elif overlap >= self.THRESHOLDS['ligo']['moderate']:
                scores.append(0.5)
            else:
                scores.append(0.2)

        # CMB score
        if 'cmb' in verification and 'metrics' in verification['cmb']:
            chi2_dof = verification['cmb']['metrics']['reduced_chi_squared']
            if chi2_dof <= self.THRESHOLDS['cmb']['excellent']:
                scores.append(1.0)
            elif chi2_dof <= self.THRESHOLDS['cmb']['good']:
                scores.append(0.7)
            elif chi2_dof <= self.THRESHOLDS['cmb']['moderate']:
                scores.append(0.5)
            else:
                scores.append(0.2)

        # Particle score
        if 'particles' in verification and 'pdg' in verification['particles']:
            match_frac = verification['particles']['pdg']['match_fraction']
            if match_frac >= self.THRESHOLDS['particles']['excellent']:
                scores.append(1.0)
            elif match_frac >= self.THRESHOLDS['particles']['good']:
                scores.append(0.7)
            elif match_frac >= self.THRESHOLDS['particles']['moderate']:
                scores.append(0.5)
            else:
                scores.append(0.2)

        # Overall novelty score (average of available scores)
        if scores:
            novelty_score = np.mean(scores)
            is_novel = novelty_score >= 0.5  # At least moderate match required
        else:
            novelty_score = 0.0
            is_novel = False

        results['novelty_score'] = float(novelty_score)
        results['is_novel'] = is_novel
        results['interpretation'] = self._interpret_novelty(novelty_score, verification)
        results['recommendations'] = self._generate_recommendations(verification, is_novel)

        return results

    def _interpret_ligo(self, overlap: float) -> str:
        """Interpret LIGO overlap value."""
        if overlap >= 0.9:
            return "Excellent - strong GW-like pattern"
        elif overlap >= 0.7:
            return "Good - significant GW similarity"
        elif overlap >= 0.5:
            return "Moderate - some GW features"
        else:
            return "Weak - limited GW similarity"

    def _interpret_cmb(self, chi2_dof: float) -> str:
        """Interpret CMB χ²/DOF value."""
        if chi2_dof <= 1.5:
            return "Excellent - strong CMB-like pattern"
        elif chi2_dof <= 3.0:
            return "Good - significant CMB similarity"
        elif chi2_dof <= 5.0:
            return "Moderate - some CMB features"
        else:
            return "Weak - limited CMB similarity"

    def _interpret_particles(self, match_fraction: float) -> str:
        """Interpret particle match fraction."""
        if match_fraction >= 0.7:
            return "Excellent - strong SM-like pattern"
        elif match_fraction >= 0.5:
            return "Good - significant SM similarity"
        elif match_fraction >= 0.3:
            return "Moderate - some SM features"
        else:
            return "Weak - limited SM similarity"

    def _interpret_novelty(self, score: float, verification: Dict) -> str:
        """Generate overall interpretation."""
        if score >= 0.7:
            return (f"NOVEL EMERGENT PHENOMENON (score: {score:.2f}): "
                   f"Exhibits strong patterns similar to real physics. This phenomenon "
                   f"shows mathematical structures analogous to empirical observations, "
                   f"strengthening the novelty claim.")
        elif score >= 0.5:
            return (f"POTENTIALLY NOVEL (score: {score:.2f}): "
                   f"Shows moderate similarity to real physics patterns. Further investigation "
                   f"recommended to validate novelty.")
        else:
            return (f"INSUFFICIENT VERIFICATION (score: {score:.2f}): "
                   f"Does not exhibit strong patterns similar to real physics. May still be "
                   f"novel if it meets other criteria (topological origin, reproducibility, etc.).")

    def _generate_recommendations(self, verification: Dict, is_novel: bool) -> List[str]:
        """Generate recommendations for next steps."""
        recommendations = []

        if is_novel:
            recommendations.append("✓ Document in EMERGENTS.md with full template including verification results")
            recommendations.append("✓ Generate whitepaper section highlighting real-world pattern matching")
            recommendations.append("✓ Update README.md if this represents a new capability")
        else:
            recommendations.append("! Run additional validation tests (reproducibility, topological specificity)")
            recommendations.append("! Check correlation with RNN parameters (|r| > 0.7)")
            recommendations.append("! Consider testing with different verification parameters")

        # Specific recommendations based on results
        if 'ligo' in verification and verification['ligo'].get('best_match', {}).get('overlap', 0) < 0.5:
            recommendations.append("→ LIGO match weak - try different waveform events or longer time series")

        if 'cmb' in verification and verification['cmb'].get('metrics', {}).get('reduced_chi_squared', 999) > 5.0:
            recommendations.append("→ CMB match weak - try different multipole ranges or spectrum types (EE/BB)")

        if 'particles' in verification and verification['particles'].get('pdg', {}).get('match_fraction', 0) < 0.3:
            recommendations.append("→ Particle match weak - check energy scale calibration or try LHC spectra")

        return recommendations

    def _prepare_for_json(self, results: Dict) -> Dict:
        """Convert torch tensors to lists for JSON serialization."""
        def convert(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            else:
                return obj

        return convert(results)


def verify_emergent_phenomenon(
    field_tensor: Optional[torch.Tensor] = None,
    vortex_energies: Optional[torch.Tensor] = None,
    phenomenon_type: str = 'auto'
) -> Dict[str, any]:
    """
    Convenience function for automated emergent verification.

    Args:
        field_tensor: Field evolution or configuration tensor
        vortex_energies: Vortex energy levels
        phenomenon_type: Type of phenomenon or 'auto' for detection

    Returns:
        Verification results with novelty assessment
    """
    verifier = EmergentVerifier()
    return verifier.verify_phenomenon(
        field_tensor=field_tensor,
        vortex_energies=vortex_energies,
        phenomenon_type=phenomenon_type
    )
