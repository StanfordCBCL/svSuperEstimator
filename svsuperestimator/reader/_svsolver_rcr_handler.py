"""This module holds the SvSolverRcrHandler class."""
from ._plain_handler import PlainHandler


class SvSolverRcrHandler(PlainHandler):
    """Handler for svSolver RCR BC data."""

    def get_rcr_data(self) -> dict:
        """Get the RCR data.

        Returns:
            rcr_data: Dict with Rp, C, Rd, Pd and t for all RCR boundary
                conditions.
        """
        bc_data = []
        i = 0
        ele_data = {}
        for j, line in enumerate(self.data.splitlines()):
            if j == 0:
                continue
            if i == 0:
                num_data = int(line)
            elif i == 1:
                ele_data["Rp"] = float(line)
            elif i == 2:
                ele_data["C"] = float(line)
            elif i == 3:
                ele_data["Rd"] = float(line)
            elif i == 4:
                t, pd = line.split()
                ele_data["t"] = [float(t)]
                ele_data["Pd"] = [float(pd)]
            elif 4 < i < 4 + num_data:
                t, pd = line.split()
                ele_data["t"].append(float(t))
                ele_data["Pd"].append(float(pd))
                if i == 3 + num_data:
                    bc_data.append(ele_data)
                    ele_data = {}
                    i = 0
                    continue
            i += 1
        return bc_data

    def set_rcr_data(self, rcr_data: dict) -> None:
        """Set the RCR data.

        Args:
            rcr_data: Dict with Rp, C, Rd, Pd and t for all RCR boundary
                conditions.
        """
        max_points = max([len(rcr["Pd"]) for rcr in rcr_data])
        self.data = f"{max_points}\n"

        for rcr in rcr_data:
            self.data += (
                f"{len(rcr['Pd'])}\n{rcr['Rp']}\n{rcr['C']}\n{rcr['Rd']}\n"
                + "\n".join(
                    [f"{t} {pd}" for t, pd in zip(rcr["t"], rcr["Pd"])]
                )
                + "\n"
            )
