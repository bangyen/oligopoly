"""REST API for the oligopoly simulation.

Routes live in :mod:`sim.api.simulate`, :mod:`sim.api.runs` and
:mod:`sim.api.heatmap`; request/response models in :mod:`sim.api.schemas`.
"""

from sim.database import get_db

from .main import app, serve

__all__ = ["app", "get_db", "serve"]
