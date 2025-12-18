# Real-World Data Verification System

**Version**: 1.0.0
**Added**: v0.1.0 (December 2025)
**Status**: Production Ready

---

## Overview

The HHmL verification system grounds emergent phenomena in empirical physics by comparing simulation outputs to real-world experimental data from:

1. **LIGO/Virgo Gravitational Waves** - Boundary resonances → GW waveforms
2. **Planck CMB** - Field fluctuations → Temperature anisotropies
3. **LHC/PDG Particles** - Excitation energies → Particle masses/spectra

This moves HHmL from pure exploration toward **testable hypotheses** against actual physical observations.

---

## 🎯 Philosophy

### What This Is

- **Analogical Comparison**: HHmL abstractions (Möbius topology, vortex quality) are **not** direct physics, but mathematical analogs
- **Pattern Matching**: Testing if emergent mathematical structures exhibit patterns similar to physical phenomena
- **Hypothesis Testing**: Falsifiable predictions about parameter-observable correlations

### What This Is NOT

- **NOT claiming HHmL models fundamental physics**
- **NOT asserting Möbius strips are spacetime**
- **NOT replacing general relativity or quantum field theory**

**Interpretation**: If HHmL vortex dynamics correlate with LIGO waveforms, it suggests emergent topological field organization shares mathematical structure with gravitational perturbations - **not** that vortices *are* black holes.

---

## 📁 Architecture

### Directory Structure

```
src/hhml/verification/
├── __init__.py              # Module exports
├── ligo.py                  # LIGO gravitational wave verification
├── cmb.py                   # CMB power spectrum verification
└── particles.py             # Particle physics verification

examples/verification/
├── verify_ligo_example.py   # LIGO example usage
├── verify_cmb_example.py    # CMB example usage
└── verify_particles_example.py  # Particle example usage

data/                        # Cached datasets (gitignored)
├── ligo/                    # LIGO strain timeseries
├── cmb/                     # Planck Cℓ spectra
└── particles/               # LHC histograms, PDG data
```

### Dependencies

Installed via `requirements.txt`:

```bash
# LIGO
gwpy>=3.0.0

# CMB
healpy>=1.16.0
camb>=1.5.0  # Optional but recommended

# Particles
uproot>=5.0.0
awkward>=2.0.0

# Utilities
astropy>=5.3.0
```

**Install**: `pip install gwpy healpy camb uproot awkward astropy`

---

## 1. LIGO Gravitational Waves

### Concept

Map HHmL boundary resonances or vortex merger dynamics to gravitational waveforms, comparing against real LIGO/Virgo detections.

**Physical Analogy**:
- **Sim**: Möbius boundary field oscillations
- **Real**: Strain h(t) from binary mergers (GW150914, GW170817, etc.)

**Metric**: Matched-filter overlap (0-1), signal-to-noise ratio

### Usage

```python
from hhml.verification.ligo import LIGOVerification
import torch

# Initialize
verifier = LIGOVerification(data_dir="data/ligo")

# Simulate field evolution [time_steps, nodes, features]
field_tensor = torch.randn(16384, 100, 2)  # 4 sec @ 4096 Hz

# Compare to GW150914
results = verifier.compare_event(
    event_name='GW150914',
    sim_strain_tensor=field_tensor,
    detector='H1',  # Hanford detector
    save_results=True
)

print(f"Overlap: {results['metrics']['overlap']:.4f}")
print(f"SNR: {results['metrics']['snr']:.2f}")
print(f"Interpretation: {results['interpretation']}")
```

### Extracting Sim Strain

The verifier automatically extracts strain from field evolution:

```python
strain = verifier.extract_sim_strain(
    field_tensor,
    boundary_indices=None,  # Use all nodes or specify boundary
    sample_rate=4096
)
# Returns: 1D numpy array, normalized to LIGO scale (~10⁻²¹)
```

**Strategy**:
1. Sum field amplitudes over boundary nodes (∝ mass quadrupole moment)
2. Compute second time derivative: h ~ d²ψ/dt²
3. Normalize to LIGO strain scale

### Known Events

Pre-configured LIGO events:
- `GW150914`: First detection (BBH merger, 29+36 M☉)
- `GW151226`: Second detection (BBH, 14+7.5 M☉)
- `GW170817`: Binary neutron star merger (multi-messenger)

### Data Sources

- **Primary**: [GWOSC](https://gwosc.org/data/) (Gravitational Wave Open Science Center)
- **Automatic**: `gwpy` fetches data directly from GWOSC servers
- **Fallback**: Synthetic chirp if `gwpy` unavailable

### Interpretation

| Overlap | Interpretation |
|---------|----------------|
| > 0.9 | Excellent waveform agreement |
| 0.7-0.9 | Good - significant similarity |
| 0.5-0.7 | Moderate - some features captured |
| 0.3-0.5 | Weak - limited agreement |
| < 0.3 | Poor - minimal similarity |

---

## 2. CMB Power Spectra

### Concept

Map HHmL internal field fluctuations to cosmic microwave background temperature anisotropies, comparing angular power spectra Cℓ vs Planck observations.

**Physical Analogy**:
- **Sim**: Lattice field projected onto sphere → angular power spectrum
- **Real**: Planck TT/EE/BB spectra from early universe

**Metric**: χ² goodness-of-fit, p-value

### Usage

```python
from hhml.verification.cmb import CMBVerification
import torch

# Initialize
verifier = CMBVerification(data_dir="data/cmb", nside=512)

# Simulate field configuration [nodes, features]
field_tensor = torch.randn(10000, 1)

# Compare to Planck TT spectrum
results = verifier.compare_planck(
    sim_field_tensor=field_tensor,
    cl_type='TT',  # Temperature auto-correlation
    lmax=2000,
    save_results=True
)

print(f"χ²: {results['metrics']['chi_squared']:.2f}")
print(f"χ²/DOF: {results['metrics']['reduced_chi_squared']:.3f}")
print(f"p-value: {results['metrics']['p_value']:.4f}")
```

### Computing Sim Cℓ

The verifier projects lattice fields to HEALPix maps and computes angular power spectra:

```python
ells, cl_sim = verifier.compute_cl_from_sim(field_tensor, nside=512)
# Returns: (multipole ℓ values, Cℓ power spectrum)
```

**Strategy**:
1. Map lattice nodes to HEALPix pixels (interpolation/downsampling)
2. Compute spherical harmonic transform: aℓm = ∫ f(θ,φ) Yℓm*(θ,φ) dΩ
3. Power spectrum: Cℓ = ⟨|aℓm|²⟩

### Spectrum Types

- `TT`: Temperature auto-correlation (primary)
- `EE`: E-mode polarization
- `BB`: B-mode polarization (gravitational waves signature)
- `TE`: Temperature-E-mode cross-correlation

### Data Sources

- **Primary**: Planck 2018/2020 Legacy Archive
- **Fiducial**: CAMB-generated ΛCDM spectra (H₀=67.4, Ωm=0.315)
- **Fallback**: Simple power-law if CAMB unavailable

### Interpretation

| χ²/DOF | Interpretation |
|--------|----------------|
| < 1.5 | Excellent spectrum match |
| 1.5-3.0 | Good - significant agreement |
| 3.0-5.0 | Moderate - some features captured |
| 5.0-10.0 | Weak - limited agreement |
| > 10.0 | Poor - minimal similarity |

**Cosmic Variance**: Even perfect models have χ²/DOF ≈ 1 due to statistical fluctuations.

---

## 3. Particle Physics

### Concept

Map HHmL emergent excitations (vortex energy levels, pruning rates) to particle masses and decay spectra, comparing against PDG values and LHC data.

**Physical Analogy**:
- **Sim**: Vortex energy eigenvalues → "masses", pruning rates → "decay widths"
- **Real**: Standard Model particle masses, LHC invariant mass histograms

**Metric**: χ² on mass spectra, fractional mass matches

### Usage

#### Compare to PDG Masses

```python
from hhml.verification.particles import ParticleVerification
import torch

# Initialize
verifier = ParticleVerification(data_dir="data/particles")

# Simulate vortex energies (GeV scale)
energies = torch.tensor([0.0005, 0.105, 1.78, 91.1, 125.3])

# Compare to Standard Model
results = verifier.compare_pdg_masses(
    sim_energies=energies,
    particle_list=['electron', 'muon', 'tau', 'Z_boson', 'Higgs'],
    tolerance=0.1  # ±10% match
)

print(f"Matched: {results['matched_particles']}/{results['total_particles']}")
for match in results['matches']:
    print(f"{match['particle']}: {match['match']} (error: {match['relative_error']:.2%})")
```

#### Compare to LHC Spectra

```python
# Compare to Higgs → 4 lepton channel
results = verifier.compare_lhc_channel(
    sim_energies=energies,
    channel='higgs_4l',  # Invariant mass histogram
    scale_factor=1.0,    # Conversion to GeV
    save_results=True
)

print(f"χ²/DOF: {results['metrics']['reduced_chi_squared']:.3f}")
print(f"KS statistic: {results['metrics']['ks_statistic']:.4f}")
```

### Known Particles (PDG 2024)

**Leptons**:
- Electron: 0.511 MeV
- Muon: 105.66 MeV
- Tau: 1.777 GeV

**Quarks**:
- Up: 2.16 MeV
- Down: 4.67 MeV
- Charm: 1.27 GeV
- Strange: 93 MeV
- Top: 172.76 GeV
- Bottom: 4.18 GeV

**Bosons**:
- W±: 80.377 GeV
- Z⁰: 91.1876 GeV
- Higgs: 125.25 GeV

### LHC Channels

Pre-configured decay channels:
- `higgs_4l`: H → ZZ* → 4ℓ (golden channel, clean)
- `Z_ee`: Z → e⁺e⁻ (calibration standard)
- `W_enu`: W → eν (transverse mass)

### Data Sources

- **Primary**: [HEPData](https://hepdata.net/) for LHC histograms (ROOT format)
- **PDG**: [Particle Data Group](https://pdg.lbl.gov/) for SM masses/widths
- **Fallback**: Synthetic Breit-Wigner peaks if data unavailable

### Interpretation

| Metric | Good | Moderate | Poor |
|--------|------|----------|------|
| PDG match fraction | > 70% | 40-70% | < 40% |
| χ²/DOF | < 2.0 | 2.0-5.0 | > 5.0 |
| KS statistic | < 0.2 | 0.2-0.6 | > 0.6 |

---

## 🔗 Integration with HHmL Training

### As RL Reward Component

Add verification metrics to reward function:

```python
from hhml.verification import LIGOVerification, CMBVerification

ligo_verifier = LIGOVerification()
cmb_verifier = CMBVerification()

# During training
for cycle in range(num_cycles):
    # ... evolve field ...

    # Compute verification metrics
    ligo_match = ligo_verifier.compare_event('GW150914', field_tensor)
    cmb_chi2 = cmb_verifier.compare_planck(field_tensor)

    # Add to reward
    reward_ligo = ligo_match['metrics']['overlap']  # 0-1
    reward_cmb = 1.0 / (1.0 + cmb_chi2['metrics']['reduced_chi_squared'])  # 0-1

    total_reward = (
        0.7 * reward_vortex_density +
        0.15 * reward_ligo +
        0.15 * reward_cmb
    )
```

### In Whitepaper Generation

Automatically include verification results:

```python
from hhml.verification import LIGOVerification, CMBVerification, ParticleVerification

# After training
verifiers = {
    'ligo': LIGOVerification(),
    'cmb': CMBVerification(),
    'particles': ParticleVerification()
}

results = {}
results['ligo'] = verifiers['ligo'].compare_event('GW150914', final_field)
results['cmb'] = verifiers['cmb'].compare_planck(final_field)
results['particles'] = verifiers['particles'].compare_pdg_masses(vortex_energies)

# Add to whitepaper JSON
whitepaper_data['verification'] = results
```

### In Live Dashboard

Display verification metrics in real-time:

```python
# In dashboard update loop
dashboard.update({
    'cycle': cycle,
    'density': vortex_density,
    'ligo_overlap': ligo_match['metrics']['overlap'],
    'cmb_chi2': cmb_chi2['metrics']['reduced_chi_squared'],
    'particle_matches': pdg_results['matched_particles']
})
```

---

## 📊 Output Files

All verifiers save results to JSON:

### LIGO
```json
{
  "event": "GW150914",
  "detector": "H1",
  "description": "First detection - Binary black hole merger",
  "metrics": {
    "overlap": 0.7234,
    "snr": 8.45,
    "mismatch": 0.2766
  },
  "interpretation": "Good match - significant waveform similarity"
}
```

### CMB
```json
{
  "spectrum_type": "TT",
  "lmax": 2000,
  "metrics": {
    "chi_squared": 2143.5,
    "dof": 1998,
    "reduced_chi_squared": 1.073,
    "p_value": 0.123
  },
  "interpretation": "Excellent spectrum match"
}
```

### Particles
```json
{
  "channel": "higgs_4l",
  "mass_range_GeV": [100.0, 150.0],
  "metrics": {
    "chi_squared": 89.3,
    "reduced_chi_squared": 0.893,
    "ks_statistic": 0.152
  },
  "interpretation": "Excellent fit - spectrum matches well"
}
```

---

## 🧪 Testing

Run example scripts:

```bash
# LIGO
python examples/verification/verify_ligo_example.py

# CMB
python examples/verification/verify_cmb_example.py

# Particles
python examples/verification/verify_particles_example.py
```

Unit tests:

```bash
pytest tests/unit/test_verification.py
```

---

## ⚠️ Limitations and Caveats

### 1. Analogical Interpretation

- HHmL is **not** a theory of gravity, cosmology, or particle physics
- Correlations indicate **mathematical similarity**, not physical identity
- Results should be interpreted as "pattern matching" exercises

### 2. Data Availability

- **LIGO**: Requires `gwpy` (heavyweight dependency)
- **CMB**: Planck data files must be downloaded separately
- **Particles**: LHC ROOT files require manual download from HEPData
- **Fallback**: All modules provide synthetic data for testing if real data unavailable

### 3. Scale Ambiguities

- Sim energies have arbitrary units → requires `scale_factor` tuning
- No first-principles mapping from lattice field to physical strain/temperature/mass
- Normalization choices affect metrics (document all choices!)

### 4. Statistical Significance

- Small sample sizes → high p-values not necessarily meaningful
- Cosmic variance limits CMB comparisons
- LHC backgrounds complicate particle comparisons

### 5. Computational Cost

- `healpy` spherical harmonic transforms scale as O(Nside² log Nside)
- CAMB cosmology calculations ~1 sec per spectrum
- LIGO waveform processing ~0.1 sec per event

---

## 📚 Further Reading

### LIGO/Gravitational Waves
- [GWOSC Tutorials](https://gwosc.org/tutorial/)
- [gwpy Documentation](https://gwpy.github.io/)
- Abbott et al., "Observation of Gravitational Waves from a Binary Black Hole Merger", PRL 116, 061102 (2016)

### CMB
- [healpy Documentation](https://healpy.readthedocs.io/)
- [CAMB Documentation](https://camb.readthedocs.io/)
- Planck Collaboration, "Planck 2018 results. VI. Cosmological parameters", A&A 641, A6 (2020)

### Particle Physics
- [uproot Documentation](https://uproot.readthedocs.io/)
- [PDG - Review of Particle Physics](https://pdg.lbl.gov/)
- [HEPData Repository](https://hepdata.net/)

---

## 🆘 Troubleshooting

### `gwpy` installation fails
```bash
# Use conda instead of pip
conda install -c conda-forge gwpy
```

### `healpy` import error
```bash
# Requires Fortran compiler
conda install -c conda-forge healpy
# OR
pip install healpy --no-binary healpy
```

### CAMB compilation issues
```bash
# Use pre-built binaries
conda install -c conda-forge camb
```

### Missing data files
- LIGO: Auto-fetched from GWOSC (requires internet)
- CMB: Download from [Planck Legacy Archive](https://pla.esac.esa.int/)
- Particles: Download from [HEPData](https://hepdata.net/) for specific analyses

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/Zynerji/HHmL/issues)
- **Contact**: [@Conceptual1](https://twitter.com/Conceptual1)
- **Documentation**: See `docs/guides/` for more

---

**Version History**:
- v1.0.0 (2025-12-17): Initial release with LIGO, CMB, and particle verification
