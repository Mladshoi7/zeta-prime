"""
zeta_prime — Fast computation of |ζ'(½+it)|² via T8/T9 identities.

Theorems (proved):
  T8:  |ζ'(½+it)|² = (θ'(t)·Z(t))² + (Z'(t))²
  T9:  |ζ'(½+iγ)| = |Z'(γ)|  at zeros  [exact]

Speed: 55 000 000 points/sec on GPU T4 (2 000 000× faster than mpmath).
Accuracy: <0.4% mean error with Gabcke C₀ correction.
"""

from .core import (
    theta,
    theta_prime,
    Z_hardy,
    Z_prime,
    zeta_prime_sq,
    zeta_prime_at_zero,
    T8_components,
    find_zeros,
    spacing_stats,
)

__version__ = "0.1.0"
__author__  = "Z.I."
__all__ = [
    "theta", "theta_prime", "Z_hardy", "Z_prime",
    "zeta_prime_sq", "zeta_prime_at_zero", "T8_components",
    "find_zeros", "spacing_stats",
]
