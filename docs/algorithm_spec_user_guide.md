# Understanding an AlgorithmSpec

## What an AlgorithmSpec is

An AlgorithmSpec is the shared plan for one learning package. It tells the
Theory, Notebook, Experiment, and Animation tools what they must explain or
implement consistently.

It is not a finished lesson, runnable experiment, or video. Editing the plan
does not immediately rewrite an already generated module. After the plan
changes, affected modules must be regenerated or reviewed again.

## The main fields

| Field | Plain-language meaning | Example |
|---|---|---|
| Name | The learner-facing algorithm name. | Double Q-Learning |
| Category | The family the algorithm belongs to. | Value-based control |
| Summary | A short explanation of what the algorithm is. | Uses two value tables to reduce maximization bias. |
| Objective | What the algorithm is trying to learn or optimize. | Learn a policy while reducing value overestimation. |
| Assumptions | Conditions that must be true for the material to be valid. | States and actions are finite and discrete. |
| States | Information describing the current situation. | The agent's current FrozenLake cell. |
| Actions | Choices available to the agent. | Left, Down, Right, Up. |
| Inputs | Information supplied before or during a run. | Environment, episode count, random seed. |
| Outputs | Results produced by the algorithm. | Value tables, policy, reward history. |
| Core equations | Mathematical rules the modules must preserve. | The Double Q update equation. |
| Pseudocode | Ordered, human-readable algorithm steps. | Select an action, observe a transition, update one table. |
| Supported environments | Environments the implementation promises to support. | FrozenLake-v1. |

## What the Settings table does

The Settings table describes experiment controls. It is a design
specification used by the Experiment Agent; it is not a live variable editor
for an experiment that has already been generated.

| Column | Meaning |
|---|---|
| Setting | Stable technical name used by the experiment, such as `epsilon`. |
| Starting value | Value used when the learner first opens the experiment. |
| What it changes | Plain-language explanation shown to the Provider and used by the Agent. |
| Lowest value | Smallest value the learner should be allowed to enter. |
| Highest value | Largest value the learner should be allowed to enter. |
| Change per click | Increment used by a number control or slider. |
| Allowed values | Fixed choices when the setting is not a free number. |

Example:

| Setting | Starting value | What it changes | Lowest value | Highest value | Change per click |
|---|---:|---|---:|---:|---:|
| epsilon | 0.10 | How often the agent explores instead of choosing its current best action. | 0.00 | 1.00 | 0.05 |
| gamma | 0.99 | How strongly future rewards affect the current update. | 0.00 | 1.00 | 0.01 |

The normal sequence is:

1. The Provider reviews or changes the Settings table.
2. The confirmed AlgorithmSpec is saved.
3. The Experiment Agent reads those settings and creates `experiment.py`.
4. The installed experiment turns its own parameter definitions into learner
   controls such as number inputs, checkboxes, or choice lists.

If a Provider changes the Settings table after Experiment has been generated,
the existing experiment does not change silently. It must be regenerated or
replaced and reviewed again.

## Other tables

### Symbol and Meaning

This is an optional legend displayed beside an animation formula. A row such
as `gamma — discount factor` explains notation to the learner. It does not
create a Python variable.

### Step title, Explanation, and Formula

Each row becomes one learner-facing derivation step beside an animation. It
controls explanatory content, not video rendering.

### Storyboard scenes

Each scene records its teaching purpose, estimated duration, narration,
on-screen text, visual direction, formulas, and transition. The Creator uses
this plan to make a video outside the platform.

## Provider and Reviewer

The **Provider** prepares the learning package. The Provider uploads source
material, asks Agents to generate content, edits generated content, replaces
files, selects an animation concept, and uploads the finished MP4.

The **Reviewer** controls acceptance and publication. The Reviewer approves a
module, requests changes, rejects a draft, and installs the package after all
required modules pass review.

The names are audit identities in the current version, not login accounts or a
complete permission system. For an independent review process, use different
people for these roles.

## Reading module statuses

| Status | Meaning |
|---|---|
| Not generated | No learner-facing module exists yet. Internal safety files are hidden and cannot be approved. |
| Awaiting review | A generated or uploaded module passed technical validation and needs human review. |
| Changes requested | The Reviewer identified work that the Provider must address. |
| Approved | The current module revision is accepted and locked. |
| Validation failed | The module cannot be reviewed until a technical problem is fixed. |

## Evidence and warnings

Verified evidence is an exact excerpt found in the uploaded source. It may
remain in the source language even when the generated fields are English.
Platform verification is authoritative. Agent cautions are additional notes
and do not replace source verification or human review.

## A simple review checklist

- Can a learner understand the Summary and Objective?
- Do Inputs and Outputs describe what goes into and comes out of a run?
- Are the equations and pseudocode consistent with the source?
- Does every adjustable setting have an understandable description and a safe
  range?
- Does the Experiment use the same settings named in the AlgorithmSpec?
- Are evidence excerpts traceable to the uploaded source?
- Are Theory, Notebook, Experiment, and Animation mutually consistent?
