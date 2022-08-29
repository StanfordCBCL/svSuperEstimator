"""This module holds the SvSolverInflowHandler class."""
from __future__ import annotations

from ._plain_handler import PlainHandler


class SvSolverInflowHandler(PlainHandler):
    """Handler for svSolver inflow BC data."""

    def get_inflow_data(self) -> dict[str, list]:
        """Get the inflow data.

        Returns:
            inflow_data: Dict with arrays t and Q for time step and flow,
                respectively.
        """
        time, flow = [], []
        for line in self.data.splitlines():
            t, q = line.split()
            time.append(float(t))
            flow.append(-float(q))
        return {"t": time, "Q": flow}
