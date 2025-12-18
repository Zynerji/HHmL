# Contributing to HHmL

First off, thank you for considering contributing to HHmL! This project thrives on community contributions, whether they're bug reports, feature requests, documentation improvements, or code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

---

## Code of Conduct

This project adheres to a simple code of conduct:

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful, actionable feedback
- **Be collaborative**: Work together to improve the project
- **Be patient**: Remember that contributors have varying skill levels

Unacceptable behavior will not be tolerated and may result in removal from the project.

---

## Getting Started

### Prerequisites

- Python 3.12+
- Git
- (Optional) CUDA 12.1+ for GPU development
- (Optional) Docker for containerized development

### Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/HHmL.git
   cd HHmL
   ```

2. **Create a virtual environment**

   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install in editable mode with dev dependencies**

   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**

   ```bash
   pre-commit install
   ```

5. **Verify setup**

   ```bash
   pytest tests/
   ```

---

## Development Workflow

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

**Branch naming conventions:**
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions/fixes
- `perf/` - Performance improvements

### 2. Make Your Changes

- Write clean, well-documented code
- Follow the [Code Standards](#code-standards)
- Add tests for new functionality
- Update documentation as needed

### 3. Test Your Changes

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_mobius.py

# Run with coverage
pytest --cov=hhml --cov-report=html

# Run code quality checks
black src/ tests/
flake8 src/ tests/
mypy src/
```

### 4. Commit Your Changes

Follow the **Conventional Commits** specification:

```bash
git add .
git commit -m "feat: add vortex persistence tracking"
# or
git commit -m "fix: resolve boundary condition bug in Möbius parameterization"
# or
git commit -m "docs: update installation guide with Docker instructions"
```

**Commit types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `style:` - Code style (formatting, missing semi-colons, etc.)
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

### 5. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- **Clear title** following conventional commit format
- **Description** of what changed and why
- **Related issues** (if applicable)
- **Testing** details

---

## Code Standards

### Python Style Guide

We follow **PEP 8** with some modifications:

- **Line length**: 100 characters (enforced by Black)
- **Formatting**: Black (automatic formatting)
- **Linting**: Flake8
- **Type hints**: Preferred but not required
- **Docstrings**: Required for all public functions/classes

### Code Formatting

```bash
# Auto-format code
black src/ tests/

# Check formatting without changes
black --check src/ tests/
```

### Linting

```bash
# Run linter
flake8 src/ tests/

# Configuration in pyproject.toml and .flake8
```

### Type Checking

```bash
# Run type checker
mypy src/

# Configuration in pyproject.toml
```

### Example Code

```python
"""Module for Möbius strip parameterization."""

from typing import Tuple
import torch


def parameterize_mobius(
    u: torch.Tensor,
    v: torch.Tensor,
    radius: float = 1.0,
    width: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Parameterize a Möbius strip in 3D space.

    Args:
        u: Angular parameter in [0, 2π), shape (N,)
        v: Radial parameter in [-width, width], shape (M,)
        radius: Strip radius (default: 1.0)
        width: Half-width of strip (default: 0.5)

    Returns:
        Tuple of (x, y, z) coordinates, each of shape (N, M)

    Example:
        >>> u = torch.linspace(0, 2*torch.pi, 100)
        >>> v = torch.linspace(-0.5, 0.5, 50)
        >>> x, y, z = parameterize_mobius(u, v)
    """
    # Implementation
    ...
```

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Target coverage**: 90%+
- **Critical paths**: 100% coverage required

### Test Structure

```
tests/
├── unit/               # Fast, isolated tests
│   ├── test_mobius.py
│   └── test_resonance.py
├── integration/        # Multi-component tests
│   ├── test_training_loop.py
│   └── test_checkpoint_recovery.py
└── benchmarks/         # Performance tests
    └── test_scaling.py
```

### Writing Tests

```python
import pytest
import torch
from hhml.core.mobius import MobiusStrip


class TestMobiusStrip:
    """Test suite for MobiusStrip class."""

    @pytest.fixture
    def strip(self):
        """Create a standard test Möbius strip."""
        return MobiusStrip(nodes=1000, radius=1.0, width=0.5)

    def test_initialization(self, strip):
        """Test correct initialization of Möbius strip."""
        assert strip.nodes == 1000
        assert strip.radius == 1.0
        assert strip.width == 0.5

    def test_parameterization(self, strip):
        """Test Möbius strip parameterization correctness."""
        x, y, z = strip.get_coordinates()
        assert x.shape == (1000, 3)
        # Add more specific assertions

    @pytest.mark.gpu
    def test_gpu_acceleration(self, strip):
        """Test GPU computation (requires CUDA)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        strip.to('cuda')
        # Test GPU-specific functionality
```

### Running Tests

```bash
# All tests
pytest

# Specific marker
pytest -m unit
pytest -m gpu

# Parallel execution
pytest -n auto

# Verbose output
pytest -v

# With coverage
pytest --cov=hhml --cov-report=term-missing
```

---

## Documentation

### Docstring Format

We use **Google-style docstrings**:

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Brief description (one line).

    Longer description if needed. Can span multiple lines.
    Explain the purpose, behavior, and any important details.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input is provided
        RuntimeError: When operation fails

    Example:
        >>> result = function_name(value1, value2)
        >>> print(result)
        Expected output

    Note:
        Additional notes or warnings if applicable.
    """
```

### Documentation Updates

When adding features, update:

1. **Inline docstrings** in code
2. **README.md** if user-facing
3. **docs/** directory for detailed guides
4. **examples/** directory with usage examples

### Building Documentation

```bash
# Install doc dependencies
pip install -e ".[docs]"

# Build HTML docs
cd docs/
make html

# View locally
open _build/html/index.html
```

---

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows style guidelines (Black, Flake8, MyPy pass)
- [ ] All tests pass (`pytest`)
- [ ] Test coverage ≥80% for new code
- [ ] Documentation updated (docstrings, README, docs/)
- [ ] CHANGELOG.md updated with changes
- [ ] Commit messages follow Conventional Commits
- [ ] PR description is clear and complete
- [ ] Related issues are linked

### Pull Request Template

```markdown
## Description

Brief description of changes.

## Motivation and Context

Why is this change needed? What problem does it solve?

Fixes #(issue number)

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing

Describe the tests you ran and how to reproduce them.

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code where needed
- [ ] I have updated the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix/feature works
- [ ] New and existing tests pass locally
- [ ] I have updated CHANGELOG.md
```

### Review Process

1. **Automated checks** run on PR (CI/CD)
2. **Maintainer review** (usually within 48 hours)
3. **Discussion and iteration** if changes requested
4. **Approval and merge** when ready

---

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Twitter**: [@Conceptual1](https://twitter.com/Conceptual1)

### Getting Help

- **Documentation**: Check `docs/` directory first
- **Examples**: Review `examples/` directory
- **Issues**: Search existing issues before creating new one
- **Discussions**: Ask in GitHub Discussions for general help

### Reporting Bugs

Use the **bug report template**:

```markdown
**Describe the bug**
A clear and concise description of the bug.

**To Reproduce**
Steps to reproduce:
1. ...
2. ...

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.12.1]
- HHmL version: [e.g., 0.1.0]
- CUDA version (if applicable): [e.g., 12.1]

**Additional context**
Any other context, logs, or screenshots.
```

### Feature Requests

Use the **feature request template**:

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots.
```

---

## Recognition

Contributors will be recognized in:

- **CHANGELOG.md**: Your contributions listed by version
- **README.md**: Acknowledgments section
- **AUTHORS**: Full contributor list

Significant contributors may be invited to join the core team.

---

## Questions?

If you have questions about contributing, feel free to:

- Open a GitHub Discussion
- Reach out on Twitter [@Conceptual1](https://twitter.com/Conceptual1)
- Create an issue with the `question` label

Thank you for contributing to HHmL! 🎭
