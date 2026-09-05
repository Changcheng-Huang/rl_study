# Algorithm teaching package format (schema versions 1 and 2)

An algorithm package is a ZIP containing `manifest.json` and a non-empty
Markdown theory file. Animation, notebook, and executable experiment content
are optional.

Schema v1 remains supported for backward compatibility and can still be
validated and installed directly. New generated content uses Schema v2 and
must pass the draft review workflow before installation.

```text
algorithm/
├── manifest.json
├── theory.md
├── animation.mp4
├── notebook.ipynb
└── experiment.py
```

The ZIP may contain these files directly or wrap them in one top-level folder.

## Minimal manifest

```json
{
  "schema_version": 1,
  "id": "example-algorithm",
  "name": "Example Algorithm",
  "version": "1.0.0",
  "summary": "A short teaching summary.",
  "category": "value-based",
  "theory": {"file": "theory.md"}
}
```

## Schema v2 AlgorithmSpec

Schema v2 uses `manifest.json` as the single source of truth. Theory, Notebook,
and Experiment are required by the v2 MVP; Animation remains optional.

Newly generated packages may add two backward-compatible presentation files:

```json
{
  "modules": {
    "theory": {
      "file": "theory.md",
      "presentation_file": "theory.presentation.json"
    },
    "notebook": {
      "file": "notebook.ipynb",
      "requirements": [{"package": "numpy>=2.2", "import": "numpy"}],
      "validation": "static-only-not-executed"
    },
    "experiment": {
      "module": "experiment.py",
      "requirements": [],
      "spec_file": "experiment_spec.json"
    }
  }
}
```

The Theory sidecar drives the shared Concept, Math, Pseudocode, and Checkpoint
tabs. `experiment_spec.json` is produced by an isolated `get_spec()` call during
publication so opening the installed lab never executes package code. Legacy v2
packages without either sidecar remain supported.

```json
{
  "schema_version": 2,
  "id": "example-algorithm",
  "name": "Example Algorithm",
  "version": "1.0.0",
  "summary": "A reviewed teaching module.",
  "category": "value-based",
  "sources": [
    {
      "type": "file",
      "path": "sources/source.md",
      "sha256": "<64 lowercase hex characters>",
      "name": "source.md"
    },
    {"type": "url", "url": "https://example.com/reference"}
  ],
  "algorithm": {
    "objective": "Learn an action-value function.",
    "assumptions": ["Finite state space"],
    "inputs": ["Environment"],
    "outputs": ["Policy"],
    "states": ["Environment state"],
    "actions": ["Available action"],
    "hyperparameters": {"gamma": {"default": 0.99}},
    "core_equations": ["Q(s,a) = E[G_t]"],
    "pseudocode": ["Collect experience", "Update values"],
    "supported_environments": ["FrozenLake-v1"],
    "experiment_design": {
      "provenance": {
        "type": "platform_preset",
        "preset_id": "frozen-lake-4x4-v1",
        "label": "Standard 4×4 FrozenLake",
        "note": "Platform-supplied teaching scenario, not a source quotation."
      },
      "task": {
        "mission": "Reach Goal without entering a Hole.",
        "dynamics": ["Deterministic grid movement."],
        "rewards": ["Goal: +1; otherwise: 0."]
      },
      "environment_map": {
        "kind": "grid",
        "layout": ["SFFF", "FHFH", "FFFH", "HFFG"],
        "legend": {
          "S": {"label": "START", "role": "start", "terminal": false},
          "F": {"label": "ICE", "role": "normal", "terminal": false},
          "H": {"label": "HOLE", "role": "hazard", "terminal": true},
          "G": {"label": "GOAL", "role": "goal", "terminal": true}
        },
        "actions": {
          "0": {"label": "Left", "arrow": "←"},
          "1": {"label": "Down", "arrow": "↓"},
          "2": {"label": "Right", "arrow": "→"},
          "3": {"label": "Up", "arrow": "↑"}
        }
      },
      "transition_model": {
        "kind": "deterministic_grid",
        "out_of_bounds": "stay",
        "start_symbol": "S",
        "goal_symbols": ["G"],
        "hazard_symbols": ["H"],
        "terminal_symbols": ["G", "H"],
        "step_reward": 0,
        "goal_reward": 1,
        "hazard_reward": 0
      }
    }
  },
  "modules": {
    "theory": {"file": "theory.md"},
    "notebook": {"file": "notebook.ipynb"},
    "experiment": {
      "module": "experiment.py",
      "requirements": []
    }
  },
  "generation": {
    "mode": "template",
    "generator_version": "template-v2.0",
    "generated_at": "2026-07-28T00:00:00+00:00",
    "blocking_flags": ["placeholder_content"],
    "algorithm_spec_agent": {
      "provider": "langchain-openai-compatible",
      "framework": "langchain",
      "model": "provider-model-name",
      "structured_output_method": "function_calling",
      "response_id": "run-...",
      "accepted_after_manual_review": true
    },
    "algorithm_spec_confirmation": {
      "confirmed_by": "Ada",
      "confirmed_at": "2026-07-31T00:00:00+00:00",
      "spec_sha256": "<64 lowercase hex characters>"
    },
    "module_generations": {
      "theory": [
        {
          "framework": "langchain",
          "model": "provider-model-name",
          "structured_output_method": "function_calling",
          "prompt_version": "module-agents-v1"
        }
      ]
    }
  },
  "review": {
    "modules": {
      "theory": {"status": "not_generated"},
      "notebook": {"status": "not_generated"},
      "experiment": {"status": "not_generated"}
    },
    "history": []
  }
}
```

`generation.algorithm_spec_agent` is optional. When present, it records that an
editable AlgorithmSpec suggestion came through a LangChain model adapter; it
does not mean the content was automatically approved. API keys and base URLs
are never stored in the package.

`generation.algorithm_spec_confirmation` records the professional confirmation
of the exact AlgorithmSpec snapshot. `generation.module_generations` records
successful Theory, Notebook, and Experiment Agent revisions, including model,
prompt version, response metadata, warnings, and token usage.

Supported module states are `not_generated`, `generating`, `draft`,
`validation_failed`, `awaiting_review`, `changes_requested`, `approved`, and
`installed`.

Schema v2 installation is rejected when a generation blocking flag is present
or when any core module is not approved/installed. If Animation is declared,
its review state is also required and must be approved/installed. A generic
scaffold's `placeholder_content` flag is cleared automatically after all three
core modules have validated Agent-generated revisions. Manual or mixed
completion requires Theory editing or validated Notebook/Experiment replacement
uploads plus a reviewer completion assertion. A completed module may be approved
while other modules remain scaffolds; scaffold modules and final installation
remain blocked. Human module approval is mandatory. The management UI does not
accept v2 ZIPs through the legacy direct-install control.

The `id` must use lowercase letters, numbers, and hyphens. A v2 package may
add this entry inside `modules`:

```json
{
  "animation": {
    "file": "animation.mp4",
    "concept_markdown": "Explain what the viewer should understand.",
    "formula": "Q(s,a)=...",
    "symbols": [
      {"symbol": "alpha", "meaning": "learning rate"}
    ],
    "highlights": ["First point", "Second point"],
    "viewing_flow": ["Watch the state change", "Compare it with the formula"],
    "derivation_steps": [
      {
        "title": "Return",
        "text": "Accumulate the discounted rewards.",
        "latex": "G_t = R_{t+1} + gamma G_{t+1}"
      }
    ]
  }
}
```

Animation metadata is editable independently from the MP4. Legacy derivation
steps using `name`/`content` are read as `title`/`text`; new drafts must use the
canonical fields shown above.

The corresponding `review.modules.animation` entry is required. Animation is
an uploaded, finished MP4 in this MVP; the platform does not generate it or
execute Manim source. MP4 files must be non-empty, recognizable as MP4, and no
larger than 200 MiB.

## Experiment contract

`experiment.py` must expose synchronous `get_spec()` and
`run(parameters, reporter)` functions:

```python
def get_spec():
    return {
        "parameters": {
            "episodes": {
                "type": "int",
                "default": 100,
                "min": 1,
                "max": 1000,
                "step": 1
            }
        },
        "presentation": {
            "task": {
                "mission": "Reach the goal.",
                "dynamics": ["Move one cell per step."],
                "rewards": ["The goal gives +1."]
            },
            "environment_map": {
                "kind": "grid",
                "layout": ["SF", "HG"],
                "legend": {
                    "S": {"label": "Start", "role": "start"},
                    "F": {"label": "Safe", "role": "normal"},
                    "H": {"label": "Hole", "role": "hazard", "terminal": True},
                    "G": {"label": "Goal", "role": "goal", "terminal": True}
                },
                "actions": {
                    "0": {"label": "Right", "arrow": "→"}
                }
            }
        }
    }


def run(parameters, reporter):
    rewards = []
    for episode in range(parameters["episodes"]):
        reward = 0.0
        rewards.append(reward)
        reporter.progress(episode + 1, parameters["episodes"])
        reporter.metric("reward", reward, step=episode)
    return {
        "metrics": {"reward": rewards},
        "summary": {"final_reward": rewards[-1]},
        "artifacts": [],
        "views": {
            "policy_grid": {
                "state_values": [0.0, 0.5, None, 1.0],
                "best_actions": [0, 0, None, None]
            }
        }
    }
```

Supported parameter types are `int`, `float`, `bool`, `string`, and `choice`.
Result metrics are named numeric lists. Summary values are JSON scalars.
Artifacts use `image`, `table`, `text`, or `video` and must reference a
relative path inside the installed package.
`presentation` and `views.policy_grid` are optional. A declared grid must be
rectangular and no larger than 20 by 20; policy values and actions are validated
again in the parent process before rendering.

## Build

Use the project virtual environment:

```bash
.venv/bin/python tools/build_algorithm_package.py path/to/algorithm
```

The validated archive is written to `dist/<id>-<version>.zip`. Missing
experiment dependencies are warnings; package code is never executed while
validating or building the ZIP.

## Repository example

Build the included Monte Carlo Control package:

```bash
.venv/bin/python tools/build_algorithm_package.py \
  algorithm_packages/examples/monte_carlo_control
```

Then start the site with `.venv/bin/streamlit run web/app.py`, open
`Manage Algorithms`, and upload `dist/monte-carlo-control-1.0.0.zip`.

For the v2 workflow, open **Create v2 Draft**, select the Monte Carlo Control
preset, generate the draft, approve its three core modules (plus Animation when
one was added), and install it from **Review Drafts**.

An incorrectly installed package can be removed from the same page. Removal
moves the package to `algorithm_packages/.trash/` so the local files remain
recoverable.
