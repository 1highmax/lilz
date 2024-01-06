from __future__ import annotations

from typing import TYPE_CHECKING

import math
import numpy as np

from assignment.tensor_networks import (
    create_zero_state,
    create_nearest_neighbor_two_qubit_gate,
    apply_single_qubit_gate,
    apply_two_qubit_gate,
    get_amplitude,
    measure_all,
    simulate_ghz_state
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

MAX_QUBITS = 5
RANDOM_SEED = 42
RANDOM_ITERATIONS = 10
SHOTS = 8192


def psi_to_MPS(psi, d=2):
    """ Decomposes a state into an MPS with physical dimension d at each site.
        Length of psi and the bond dimension must have the relation
        psi_length = d^num_sites

    Args:
        psi: state vector
        d: physical dimension at each site

    Returns:
        MPS: list of rank-3 tensors corresponding to each site
    """
    MPS = []

    # Check if decompose-able (psi_length = d^num_sites)
    # Checks if float from logarithm is not close to an integer (floor or ceiling)
    # The machine error from logarithms is too much to do this an easier way
    if math.isclose(np.log(len(psi))/np.log(d), math.floor(np.log(len(psi))/np.log(d))):
        num_sites = math.floor(np.log(len(psi))/np.log(d))
    elif math.isclose(np.log(len(psi))/np.log(d), math.ceil(np.log(len(psi))/np.log(d))):
        num_sites = math.ceil(np.log(len(psi))/np.log(d))
    else:
        raise NameError('State not decompose-able into qudits for given d')

    tensor = psi
    # Iteration starting at 1 for the exponential
    for i in range(1, num_sites+1):
        if i == 1:
            tensor = np.reshape(tensor, (d, len(psi)//d))
            U, S_vector, V = np.linalg.svd(tensor, full_matrices=0)
            site_tensor = np.expand_dims(U, 0)

            # Left bond, right bond, phys dim
            # site_tensor = np.transpose(site_tensor, (0, 2, 1))
            MPS.append(site_tensor)

            left_bond = len(S_vector)

        else:
            tensor = np.reshape(tensor, (d*left_bond, len(psi)//(d**i)))
            U, S_vector, V = np.linalg.svd(tensor, full_matrices=0)

            if i == num_sites+1:
                right_bond = 1
            else:
                right_bond = U.shape[1]
            site_tensor = np.reshape(U, (left_bond, d, right_bond))
            left_bond = len(S_vector)
            MPS.append(site_tensor)

        tensor = np.diag(S_vector) @ V

    return MPS


def test_zero_state() -> None:
    for num_qubits in range(1, MAX_QUBITS + 1):
        print(f"Creating zero state for {num_qubits} qubits")
        answer = create_zero_state(num_qubits)            
        assert len(answer) == num_qubits, f"Expected {num_qubits} qubits, got {len(answer)}"
        for tensor in answer:
            assert tensor.ndim == 3, f"Expected rank-3 tensor, got {tensor.ndim}"
            assert tensor[0][0][0] == 1, f"Expected element0 to be 1, got {tensor[0][0][0]}"
            assert tensor[0][1][0] == 0, f"Expected element1 to be 0, got {tensor[0][1][0]}"


def test_nearest_neighbor_two_qubit_gate() -> None:
    SWAP_matrix = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.complex128)
    print(f"Creating two-qubit gate with matrix\n {SWAP_matrix}")
    answer = create_nearest_neighbor_two_qubit_gate(SWAP_matrix)
    assert answer.ndim == 4, f"Expected rank-4 tensor, got {answer.ndim}"


def test_apply_single_qubit_gate() -> None:
    X = np.array([[0, 1],
                  [1, 0]], dtype=np.complex128)

    answer = create_zero_state(num_qubits=1)
    apply_single_qubit_gate(answer, X, site=0)
    assert answer[0].ndim == 3, f"Expected rank-3 tensor, got {answer[0].ndim}"
    assert answer[0][0][0][0] == 0, f"Expected element0 to be 0, got {answer[0][0][0][0]}"
    assert answer[0][0][1][0] == 1, f"Expected element1 to be 1, got {answer[0][0][1][0]}"


def test_apply_two_qubit_gate() -> None:
    X = np.array([[0, 1],
                  [1, 0]], dtype=np.complex128)
    SWAP_matrix = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]], dtype=np.complex128)

    answer = create_zero_state(num_qubits=2)
    apply_single_qubit_gate(answer, X, site=0)
    SWAP_tensor = create_nearest_neighbor_two_qubit_gate(SWAP_matrix)
    apply_two_qubit_gate(answer, SWAP_tensor, site0=0, site1=1)
    assert answer[0].ndim == 3, f"Expected rank-3 tensor, got {answer[0].ndim}"
    assert answer[0][0][0][0] == 1, f"Expected element0 to be 1, got {answer[0][0][0][0]}"
    assert answer[0][0][1][0] == 0, f"Expected element1 to be 0, got {answer[0][0][1][0]}"
    assert answer[1][0][0][0] == 0, f"Expected element0 to be 0, got {answer[1][0][0][0]}"
    assert answer[1][0][1][0] == 1, f"Expected element1 to be 1, got {answer[1][0][1][0]}"


def test_get_amplitude() -> None:
    gen = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for num_qubits in range(1, MAX_QUBITS + 1):
        for _ in range(RANDOM_ITERATIONS):
            state = gen.random((2**num_qubits,))
            state /= np.linalg.norm(state)

            print(f"Testing get_amplitude for {num_qubits} qubits, with state\n{state}")

            state_dd = psi_to_MPS(state)

            for idx in range(2**num_qubits):
                bitstring = format(idx, f"0{num_qubits}b")
                answer = get_amplitude(state_dd, bitstring)
                assert answer.ndim == 0
                solution = state[idx]
                assert np.allclose(solution, answer), f"""
                Expected amplitude {solution} for bitstring {bitstring}, got {answer}.
                An error here is most likely caused by relying on the wrong qubit ordering.
                """


def test_measure_all() -> None:
    gen = np.random.Generator(np.random.PCG64(RANDOM_SEED))
    for num_qubits in range(1, MAX_QUBITS + 1):
        for _ in range(RANDOM_ITERATIONS):
            state = gen.random((2**num_qubits,))
            state /= np.linalg.norm(state)

            print(f"Testing measure_all for {num_qubits} qubits, with state\n{state}")

            state_dd = psi_to_MPS(state)

            answer = measure_all(state_dd, SHOTS)
            
            assert np.allclose(sum(answer.values()), SHOTS), f"""
            Expected {SHOTS} shots in resulting dictionary, got {sum(answer.values())}.
            """

            for key in answer:
                assert np.allclose(answer[key] / SHOTS, state[int(key, 2)] ** 2, atol=0.1), f"""
                Expected probability {state[int(key, 2)] ** 2} for bitstring {key}, got {answer[key] / SHOTS}.
                An error here is most likely caused by relying on the wrong qubit ordering.
                """


def test_simulate_ghz_state() -> None:
    for num_qubits in range(2, MAX_QUBITS + 1):
        print(f"Testing simulate_ghz_state for {num_qubits} qubits")

        answer = simulate_ghz_state(num_qubits, SHOTS)

        assert len(answer) == 2, f"""
        Expected 2 entries in the answer, got {len(answer)}.
        A proper GHZ state only has two non-zero amplitudes.
        """

        assert "0" * num_qubits in answer, f"""
        Expected bitstring {"0" * num_qubits} in the answer, got {answer}.
        """
        assert "1" * num_qubits in answer, f"""
        Expected bitstring {"1" * num_qubits} in the answer, got {answer}.
        """

        assert np.allclose(answer["0" * num_qubits] / SHOTS, 0.5, atol=0.1), f"""
        Expected probability 0.5 for bitstring {"0" * num_qubits}, got {answer["0" * num_qubits] / SHOTS}.
        """
        assert np.allclose(answer["1" * num_qubits] / SHOTS, 0.5, atol=0.1), f"""
        Expected probability 0.5 for bitstring {"1" * num_qubits}, got {answer["1" * num_qubits] / SHOTS}.
        """
