import numpy as np
import json
from pathlib import Path

from wholecell_py.processes.protein_activation import (
    run_case as run_activation,
)
from wholecell_py.processes.chromosome_segregation import (
    run_case as run_segregation,
)


def test_protein_activation_moves_counts_between_states():
    result = run_activation(
        {
            "substrates": [[2, 0]],
            "inactivated_substrates": [[0, 3]],
            "stimuli": [[1.0, 1.0]],
            "extracellular_index": 0,
            "stimulus_local_indices": [0],
            "geometry_volume": 1.0,
        },
        {
            "activation_rules": ["SIG > 0"],
            "stimuli_ids": ["SIG"],
            "n_avogadro": 6.02214076e23,
        },
    )
    np.testing.assert_array_equal(result["substrates"], [[2, 3]])
    np.testing.assert_array_equal(result["inactivated_substrates"], [[0, 0]])


def test_chromosome_segregation_requires_all_guards_then_hydrolyzes_gtp():
    result = run_segregation(
        {
            "chromosome": {
                "segregated": False,
                "polymerized_regions": [10, 10],
                "supercoiled": [1, 1],
                "n_compartments": 2,
                "sequence_len": 10,
            },
            "substrates": [4, 0, 0, 5, 0],
            "enzymes": [1, 1, 1, 1, 1],
        },
        {"gtp_cost": 3},
    )
    assert result["segregated"]
    np.testing.assert_array_equal(result["substrates"], [1, 3, 3, 2, 3])


def test_all_ten_exported_matlab_activation_rules_parse_and_evaluate():
    rules_data = json.loads(
        (
            Path(__file__).parents[1] / "reference/activation_rules.json"
        ).read_text()
    )
    rules = rules_data["rules"]
    ids = rules_data["identifiers"]
    values = {name: 0.0 for name in ids}
    values.update(
        {
            "stimulus_thiolStress": 1,
            "stimulus_ironStress": 1,
            "G6P": 6,
            "PI": 21,
            "temperature": 43,
        }
    )
    # Antibiotics and gluconate are absent; both negated rule groups pass.
    scale = 6.02214076e23 / 1000.0
    stimuli = np.asarray(
        [
            [values[name] * scale if name in {"G6P", "PI"} else values[name]]
            for name in ids
        ],
        dtype=float,
    )
    raw_stimuli = [
        i for i, name in enumerate(ids) if name not in {"G6P", "PI"}
    ]
    result = run_activation(
        {
            "substrates": np.ones((len(rules), 1)),
            "inactivated_substrates": np.zeros((len(rules), 1)),
            "stimuli": stimuli,
            "extracellular_index": 0,
            "stimulus_local_indices": raw_stimuli,
            "geometry_volume": 1.0,
        },
        {
            "activation_rules": rules,
            "stimuli_ids": ids,
            "n_avogadro": 6.02214076e23,
        },
    )
    np.testing.assert_array_equal(
        result["activated_substrates"][:, 0], [True] * len(rules)
    )


def test_matlab_num2str_text_values_control_activation_comparison():
    result = run_activation(
        {
            "substrates": [[1]],
            "inactivated_substrates": [[0]],
            "stimuli": [[6.00004]],
            "extracellular_index": 0,
            "stimulus_local_indices": [0],
            "geometry_volume": 1.0,
            "stimulus_text_values": ["6"],
        },
        {
            "activation_rules": ["  SIG > 5  "],
            "stimuli_ids": ["SIG"],
            "n_avogadro": 6.02214076e23,
        },
    )
    assert result["activated_substrates"][0, 0]
