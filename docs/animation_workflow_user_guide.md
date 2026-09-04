# Animation Workflow User Guide

## What this workflow produces

The platform plans an educational animation, but it does not render the final
video. The result of the planning steps is an English storyboard and Creator
Kit. A video creator then uses those materials in Manim, After Effects, Blender,
PowerPoint, or another production tool and uploads the finished MP4.

The complete flow is:

`Generate three concepts → Select one concept → Build one storyboard → Produce the video externally → Upload the MP4 → Review`

## Who performs each step

- The **Provider** generates options, selects a concept, reviews or edits the
  storyboard, produces or commissions the video, and uploads the MP4.
- The **Reviewer** watches the uploaded video, checks it against the storyboard
  and AlgorithmSpec, and then approves it or requests changes.
- The **Agent** proposes concepts and writes production guidance. It does not
  make or upload a video.

## Step 1: Generate three concepts

Choose one of the following actions:

- **Generate three concepts with Agent** asks the configured model to propose
  exactly three different teaching approaches.
- **Create three starter concepts** creates three local starter options without
  calling a model. Use this when the Planning Agent is unavailable.

Each concept includes:

- Teaching focus: the single idea learners should understand.
- Visual approach: how that idea could be shown on screen.
- Estimated duration: approximate finished-video length.
- Complexity: expected production difficulty.
- Production cost: relative effort or resource level.
- Best use case: the learner or lesson situation it suits.
- Trade-offs: what the approach does well and what it leaves out.

Generating new concepts replaces the previous set. If a storyboard already
exists, it becomes stale because it may no longer match the new concepts.

## Step 2: Select one concept

Compare all three concepts and select one. Selection is a Provider decision;
the Agent does not choose automatically.

Nothing is rendered at this step. The selected concept becomes the required
input for the storyboard. Selecting a different concept later makes the old
storyboard stale and requires a new storyboard or manual revision.

## Step 3: Build the storyboard

Choose one of the following actions:

- **Generate with Planning Agent** creates a detailed storyboard based only on
  the selected concept and confirmed AlgorithmSpec.
- **Create AlgorithmSpec Starter** creates local production guidance without a
  model call.

The storyboard contains scene titles, teaching purposes, timings, narration,
on-screen text, visual directions, formula checks, transitions, required
assets, accessibility notes, and production notes.

The **Creator Kit** is a downloadable ZIP containing the English production
brief and structured storyboard information. Give it to the person or team
making the video.

The Provider may edit the saved guidance. Editing guidance changes the plan;
it does not change an already uploaded MP4.

## Step 4: Produce and upload the MP4

Create the actual video outside the platform by following the storyboard. Then:

1. Choose the finished `.mp4` file.
2. Watch the local preview.
3. Check the concept explanation, main formula, highlights, viewing flow, and
   derivation notes shown beside the video.
4. Select **Add Animation to Draft**.

The Animation module then becomes **Awaiting review**. Uploading a video does
not approve it.

For workflow testing, a short placeholder MP4 is sufficient. For teaching
quality review, replace it with the finished video created from the selected
storyboard.

## Step 5: Review and revise

The Reviewer watches the complete video and checks:

- Does it teach the selected concept?
- Are formulas and terminology consistent with the AlgorithmSpec?
- Do narration and on-screen text use English?
- Are timing, text size, contrast, captions, and motion accessible?
- Does the video avoid unsupported claims?

If revision is required, the Reviewer selects **Needs Changes** and records a
specific reason. The Provider can then upload a replacement MP4 or revise its
metadata. After a module is approved, use **Reopen for Changes** to unlock it
before editing.

## Status guide

| Status | Meaning | Next action |
|---|---|---|
| Not started | No concepts exist. | Generate three concepts. |
| Concepts ready | Three choices exist, but none is selected. | Compare and select one. |
| Concept selected | One concept is selected. | Build the storyboard. |
| Storyboard ready | Production guidance and Creator Kit exist. | Produce and upload the MP4. |
| Storyboard stale | The AlgorithmSpec or selected concept changed. | Regenerate or revise the storyboard. |
| Awaiting review | An MP4 has been uploaded and validated. | Reviewer watches and decides. |
| Changes requested | The current video or metadata needs revision. | Provider replaces or edits it. |
| Approved | The current animation revision is accepted and locked. | Publish, or reopen it if changes are necessary. |

## Why a button may be disabled

- **Provider name required:** apply a Provider name in the role panel.
- **Planning Agent unavailable:** configure the model service or use starter
  concepts/guidance.
- **Select one concept first:** complete Step 2.
- **Upload an MP4 first:** choose a video file in Step 4.
- **Reviewer name required:** apply a Reviewer name before approving or
  requesting changes.
