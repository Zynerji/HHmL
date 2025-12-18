"""
Real-world data verification module for HHmL.

Compares simulation outputs to empirical physics data:
- LIGO gravitational waveforms
- CMB power spectra (Planck)
- Particle physics predictions (LHC, PDG)

This grounds HHmL's emergent phenomena in testable hypotheses.
"""

from .ligo import LIGOVerification, fetch_ligo_event, compute_waveform_match
from .cmb import CMBVerification, load_planck_cl, compute_cl_from_sim
from .particles import ParticleVerification, load_lhc_histogram, compare_spectra

__all__ = [
    'LIGOVerification',
    'fetch_ligo_event',
    'compute_waveform_match',
    'CMBVerification',
    'load_planck_cl',
    'compute_cl_from_sim',
    'ParticleVerification',
    'load_lhc_histogram',
    'compare_spectra',
]
