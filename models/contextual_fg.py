from pathlib import Path  # Explanation: imports Path to locate the local src package directory.
import sys  # Explanation: imports sys so this legacy module can add src to Python's import path.

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # Explanation: lets this module import the package before installation.

from kgcl_retro.models.contextual_fg import AtomFGAttention, ContextualFGGraphEncoder, KGContextFusion  # Explanation: re-exports contextual FG layers.

__all__ = ["AtomFGAttention", "ContextualFGGraphEncoder", "KGContextFusion"]  # Explanation: defines the legacy module API.
