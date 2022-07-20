"""Visualization helper functions"""


def cgs_pressure_to_mmgh(cgs_pressure):
    """Convert pressure from g/(cm s^2) to mmHg.

    Args:
        cgs_pressure: Pressure in CGS format.

    Returns:
        Pressure in mmHg.
    """
    return cgs_pressure * 0.00075006156130264


def cgs_flow_to_lh(cgs_flow):
    """Convert flow from cm^3/s to l/h.

    Args:
        cgs_flow: Flow in CGS format.

    Returns:
        Flow in l/h.
    """
    return cgs_flow * 3.6
