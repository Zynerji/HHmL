#!/usr/bin/env python3
"""
Whitepaper Generator - HHmL Scientific Report Generation
==========================================================
Generates comprehensive LaTeX whitepapers from simulation results.

Usage:
    from web_monitor.whitepaper_generator import WhitepaperGenerator

    generator = WhitepaperGenerator()
    generator.generate_from_results(
        results_file='test_cases/multi_strip/results/training_20251216_212826.json',
        test_name='multi_strip_tokamak'
    )

Output:
    test_cases/[test_name]/whitepapers/[test_name]_YYYYMMDD_HHMMSS.pdf

Author: HHmL Framework
Date: 2025-12-16
"""

import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict


class WhitepaperGenerator:
    """
    Generates scientific whitepapers from HHmL simulation results.

    Features:
    - Reads JSON results from test_cases/[test_name]/results/
    - Generates LaTeX with proper mathematical notation
    - Saves to test_cases/[test_name]/whitepapers/
    - Descriptive filenames with timestamp
    """

    def __init__(self, prefer_markdown: bool = False):
        self.prefer_markdown = prefer_markdown

    def generate_from_results(self, results_file: str, test_name: str = None) -> str:
        """
        Generate whitepaper from results JSON file.

        Args:
            results_file: Path to results JSON (e.g., test_cases/multi_strip/results/training_*.json)
            test_name: Test name for output directory (inferred from path if None)

        Returns:
            Path to generated whitepaper
        """
        results_path = Path(results_file)

        # Infer test name from path if not provided
        if test_name is None:
            # Extract from path: test_cases/TESTNAME/results/file.json
            parts = results_path.parts
            if 'test_cases' in parts:
                idx = parts.index('test_cases')
                test_name = parts[idx + 1] if idx + 1 < len(parts) else 'unknown'
            else:
                test_name = 'unknown'

        # Load results
        with open(results_path, 'r') as f:
            data = json.load(f)

        # Create output directory
        output_dir = Path(f'test_cases/{test_name}/whitepapers')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate descriptive filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{test_name}_{timestamp}"

        # Generate whitepaper
        if self.prefer_markdown or not self._check_pdflatex():
            print(f"Generating Markdown whitepaper: {filename}.md")
            return self._generate_markdown(data, output_dir, filename, test_name)
        else:
            print(f"Generating LaTeX whitepaper: {filename}.pdf")
            return self._generate_latex(data, output_dir, filename, test_name)

    def _generate_latex(self, data: Dict, output_dir: Path, filename: str, test_name: str) -> str:
        """Generate LaTeX whitepaper."""

        # Extract key metrics
        config = data.get('configuration', {})
        perf = data.get('performance', {})
        final_metrics = data.get('final_metrics', {})

        # Create comprehensive LaTeX document
        latex = r"""\documentclass[11pt,letterpaper]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{float}

\geometry{margin=1in}

\title{HHmL Simulation Report:\\""" + f"{test_name.replace('_', ' ').title()}" + r"""}
\author{Holo-Harmonic Möbius Lattice Framework}
\date{""" + datetime.now().strftime('%B %d, %Y') + r"""}

\begin{document}
\maketitle

\begin{abstract}
This report presents results from a computational simulation using the Holo-Harmonic Möbius Lattice (HHmL) framework.
The simulation explores emergent spacetime phenomena through RNN-controlled topological field configurations on Möbius strip geometries.
This is a mathematical and computational study investigating correlations between control parameters and emergent vortex dynamics.
\end{abstract}

\section{Introduction}

The HHmL framework investigates emergent phenomena in topologically non-trivial configurations by:
\begin{itemize}
    \item Utilizing Möbius strip topology for closed-loop holographic encoding
    \item Employing recurrent neural networks (RNNs) to control 19 distinct system parameters
    \item Tracking correlations between parameter space and emergent vortex configurations
    \item Operating as a fully transparent ``glass-box'' system for systematic discovery
\end{itemize}

This report documents simulation results for automated parameter optimization via reinforcement learning.

\section{Configuration}

\subsection{System Parameters}

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
Parameter & Value \\
\midrule
"""
        # Add configuration table
        latex += f"Number of Strips & {config.get('num_strips', 'N/A')} \\\\\n"
        latex += f"Nodes per Strip & {config.get('nodes_per_strip', 'N/A'):,} \\\\\n"
        latex += f"Total Nodes & {config.get('total_nodes', 'N/A'):,} \\\\\n"
        latex += f"Hidden Dimension & {config.get('hidden_dim', 'N/A')} \\\\\n"
        latex += f"Training Cycles & {config.get('num_cycles', 'N/A')} \\\\\n"
        latex += f"Device & {config.get('device', 'N/A')} \\\\\n"
        latex += f"Mode & {config.get('mode', 'N/A')} \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\caption{Simulation configuration parameters}
\end{table}

\subsection{RNN Control Architecture}

The system employs a 4-layer LSTM with """ + f"{config.get('hidden_dim', 512)}" + r""" hidden units to control 19 parameters across 6 categories:
\begin{enumerate}
    \item \textbf{Geometry (4)}: $\kappa$ (elongation), $\delta$ (triangularity), vortex target, QEC layers
    \item \textbf{Physics (4)}: damping, nonlinearity, amplitude variance, vortex seeding
    \item \textbf{Spectral (3)}: $\omega$ (helical frequency), diffusion timestep, reset strength
    \item \textbf{Sampling (3)}: sample ratio, neighbor factor, sparsity threshold
    \item \textbf{Mode Selection (2)}: sparse density, spectral weight
    \item \textbf{Extended Geometry (3)}: winding density, twist rate, cross-coupling
\end{enumerate}

This architecture enables \textit{glass-box} tracking of correlations between control parameters and emergent phenomena.

\section{Results}

\subsection{Performance Metrics}

\begin{table}[H]
\centering
\begin{tabular}{@{}ll@{}}
\toprule
Metric & Value \\
\midrule
"""
        # Add performance metrics
        latex += f"Total Time & {perf.get('total_time', 0):.1f} s ({perf.get('total_time', 0)/60:.2f} min) \\\\\n"
        latex += f"Avg Cycle Time & {perf.get('avg_cycle_time', 0):.3f} s \\\\\n"
        latex += f"Throughput & {perf.get('cycles_per_second', 0):.2f} cycles/s \\\\\n"
        latex += r"""\bottomrule
\end{tabular}
\caption{Computational performance}
\end{table}

\subsection{Vortex Dynamics}

Final vortex density: """ + f"{final_metrics.get('vortex_density', 0):.1%}" + r"""

Peak vortex density: """ + f"{final_metrics.get('peak_vortex_density', 0):.1%}" + r"""

Final reward: """ + f"{final_metrics.get('final_reward', 0):.2f}" + r"""

The simulation tracked vortex formation and stability across """ + f"{config.get('num_cycles', 0)}" + r""" training cycles,
with the RNN autonomously discovering optimal parameter configurations.

\section{Discussion}

\subsection{Key Findings}

This simulation demonstrates the feasibility of:
\begin{itemize}
    \item RNN-based discovery of parameter configurations that maintain vortex stability
    \item Correlation tracking between 19 control parameters and emergent vortex patterns
    \item Sequential learning across training sessions via checkpoint persistence
    \item Scalable sparse graph representations for large-scale (""" + f"{config.get('total_nodes', 0):,}" + r""" node) systems
\end{itemize}

\subsection{Scientific Merit}

The HHmL framework provides:
\begin{enumerate}
    \item \textbf{Reproducibility}: Complete parameter trajectories specify experiments
    \item \textbf{Transparency}: All control parameters tracked and accessible
    \item \textbf{Scalability}: Auto-adaptive sparse/dense modes for CPU through H200
    \item \textbf{Discovery Engine}: Automated exploration of parameter $\rightarrow$ phenomena correlations
\end{enumerate}

\section{Conclusion}

This report documents computational exploration of emergent vortex dynamics in Möbius strip topologies under RNN control.
The glass-box architecture enables systematic investigation of correlations between topological parameters and emergent spacetime-like structures.

\textbf{Disclaimer}: This work explores mathematical and computational models. No claims are made about physical reality or fundamental physics.
All results represent abstract topological field configurations subject to computational investigation and peer review.

\section{Data Availability}

Complete simulation results, including full parameter histories and vortex density trajectories, are available in:
\begin{verbatim}
""" + str(output_dir.parent / 'results') + r"""
\end{verbatim}

\section{Acknowledgments}

Generated by the HHmL Framework automated scientific reporting system.

\end{document}
"""

        # Write LaTeX file
        tex_path = output_dir / f"{filename}.tex"
        with open(tex_path, 'w', encoding='utf-8') as f:
            f.write(latex)

        # Compile PDF
        try:
            result = subprocess.run(
                ['pdflatex', '-output-directory', str(output_dir), str(tex_path)],
                check=True,
                capture_output=True,
                timeout=60
            )

            # Run twice for references
            subprocess.run(
                ['pdflatex', '-output-directory', str(output_dir), str(tex_path)],
                check=True,
                capture_output=True,
                timeout=60
            )

            pdf_path = output_dir / f"{filename}.pdf"
            if pdf_path.exists():
                print(f"[OK] Whitepaper generated: {pdf_path}")
                return str(pdf_path)
            else:
                raise FileNotFoundError("PDF not created")

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
            print(f"[WARNING] PDF generation failed: {e}")
            print("Falling back to Markdown...")
            return self._generate_markdown(data, output_dir, filename, test_name)

    def _generate_markdown(self, data: Dict, output_dir: Path, filename: str, test_name: str) -> str:
        """Generate Markdown whitepaper as fallback."""

        config = data.get('configuration', {})
        perf = data.get('performance', {})
        final_metrics = data.get('final_metrics', {})

        markdown = f"""# HHmL Simulation Report: {test_name.replace('_', ' ').title()}

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Framework**: Holo-Harmonic Möbius Lattice (HHmL)

---

## Abstract

This report presents results from a computational simulation using the HHmL framework.
The simulation explores emergent spacetime phenomena through RNN-controlled topological
field configurations on Möbius strip geometries.

---

## Configuration

| Parameter | Value |
|-----------|-------|
| Strips | {config.get('num_strips', 'N/A')} |
| Nodes per Strip | {config.get('nodes_per_strip', 'N/A'):,} |
| Total Nodes | {config.get('total_nodes', 'N/A'):,} |
| Hidden Dimension | {config.get('hidden_dim', 'N/A')} |
| Training Cycles | {config.get('num_cycles', 'N/A')} |
| Device | {config.get('device', 'N/A')} |
| Mode | {config.get('mode', 'N/A')} |

## Results

### Performance

- **Total Time**: {perf.get('total_time', 0):.1f}s ({perf.get('total_time', 0)/60:.2f} min)
- **Avg Cycle Time**: {perf.get('avg_cycle_time', 0):.3f}s
- **Throughput**: {perf.get('cycles_per_second', 0):.2f} cycles/s

### Vortex Dynamics

- **Final Vortex Density**: {final_metrics.get('vortex_density', 0):.1%}
- **Peak Vortex Density**: {final_metrics.get('peak_vortex_density', 0):.1%}
- **Final Reward**: {final_metrics.get('final_reward', 0):.2f}

---

## Conclusion

This simulation demonstrates RNN-based discovery of parameter configurations that maintain
vortex stability in Möbius strip topologies.

**Disclaimer**: This work explores mathematical models only. No claims about physical reality.

---

*Report generated by HHmL Framework*
"""

        md_path = output_dir / f"{filename}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"[OK] Markdown report generated: {md_path}")
        return str(md_path)

    def _check_pdflatex(self) -> bool:
        """Check if pdflatex is available."""
        try:
            subprocess.run(
                ['pdflatex', '--version'],
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return False


if __name__ == "__main__":
    # Example usage
    generator = WhitepaperGenerator()

    # Find most recent results file
    import glob
    results_files = glob.glob('test_cases/multi_strip/results/training_*.json')
    if results_files:
        latest = max(results_files, key=lambda p: Path(p).stat().st_mtime)
        print(f"Generating whitepaper from: {latest}")
        generator.generate_from_results(latest)
    else:
        print("No results files found. Run a simulation first.")
