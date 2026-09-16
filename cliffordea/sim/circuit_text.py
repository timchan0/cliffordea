"""Text transformations for SymFT/Clifft simulation circuits."""

import re


def swap_t_and_s_gates(circuit: str) -> str:
    """Swap T and S instructions in a circuit string.

    :param circuit: A SymFT/Clifft circuit using T and/or S instructions.
    :return: Circuit with ``T``/``T_DAG`` and ``S``/``S_DAG`` exchanged.
    """
    gate_swaps = {
        "T": "S",
        "T_DAG": "S_DAG",
        "S": "T",
        "S_DAG": "T_DAG",
    }
    gate_pattern = re.compile(
        r"(?m)^([ \t]*)(T_DAG|S_DAG|T|S)(?=[ \t\r\n]|$)"
    )
    return gate_pattern.sub(
        lambda match: f"{match[1]}{gate_swaps[match[2]]}",
        circuit,
    )


def replace_noise_level(
    circuit: str,
    old_noise_level: float,
    new_noise_level: float,
) -> str:
    """Replace a hardcoded noise level in a circuit string.

    :param circuit: Circuit containing ``old_noise_level``.
    :param old_noise_level: Noise level encoded in the input circuit.
    :param new_noise_level: Noise level to encode in the output circuit.
    :return: Circuit with every textual occurrence of the old level replaced.
    """
    return circuit.replace(str(old_noise_level), str(new_noise_level))
