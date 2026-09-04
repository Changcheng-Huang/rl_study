import streamlit as st
from .cliff_walking import run as run_cliff
from .frozen_lake import run as run_frozen
from .imported import render_imported_experiment
from algorithm_registry.integration import imported_algorithms

def run_experiments_module():
    st.header('Interactive RL Laboratory')
    st.sidebar.markdown('## Select Experiment')

    EXP_MAP = {
        'Cliff Walking': run_cliff,
        'Frozen Lake': run_frozen,
    }
    for imported in imported_algorithms():
        if imported.manifest.experiment is None:
            continue
        label = imported.manifest.name
        if label in EXP_MAP:
            label = f"{label} (Imported)"
        algorithm_id = imported.manifest.algorithm_id
        EXP_MAP[label] = (
            lambda selected_id=algorithm_id: render_imported_experiment(selected_id)
        )

    default = st.session_state.get('exp_type', 'Cliff Walking')
    options = list(EXP_MAP.keys())
    if default not in options:
        default = options[0]

    exp_type = st.sidebar.radio('Environment:', options, index=options.index(default), key='exp_type')
    st.sidebar.divider()

    EXP_MAP[exp_type]()
