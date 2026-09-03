# EEPY Development Roadmap

## Product Direction

EEPY is being developed as software for an in-vehicle edge device.

Current development hardware:

- laptop webcam as the driver-facing camera
- laptop microphone as the microphone
- laptop speakers as the local alert/audio output
- local Python application as the edge software

Future hardware may replace those components, but the core software architecture should remain usable.

Core principle:

> Safety-critical fatigue detection and immediate intervention must work locally. AI capabilities are optional enhancements.

---

## 1A. Calibration Eligibility / Hard Frame Rejection

### Status

Implemented as a per-frame technical-quality gate. Frames are classified as `valid`,
`degraded`, or `rejected`, with explicit reason codes exposed in monitoring state.

### Goal

Reject individual frames that are not technically trustworthy enough to contribute to
passive personalised calibration.

### Requirements

Reject a candidate frame when there is:

- no usable face
- an invalid face box or frame size
- malformed, non-finite, collapsed, or clearly implausible landmark geometry
- a required eye, mouth, nose, or chin landmark outside the frame
- invalid or non-finite EAR or MAR measurements
- extreme head pose where detectable
- another technical condition that makes the measurement unreliable

Mark a frame as degraded, rather than rejected, when there is:

- face-detection confidence below the preferred calibration level
- a face that occupies less than the preferred proportion of the frame
- a substantially clipped face box while required landmarks remain visible
- moderate head pose
- unavailable or non-finite head-pose estimation with otherwise usable landmarks

Valid and degraded frames may enter the future rolling awake-state gate. Rejected frames
must not. This prevents ordinary webcam noise or a non-ideal camera position from making
calibration unnecessarily impossible while still excluding structurally unusable data.

Face size, visibility, and landmark geometry are evaluated relative to frame or face
dimensions rather than fixed pixel counts. EAR and MAR remain dimensionless ratios, and
unusual but finite values are not rejected merely for differing from a population norm.
Head pose is evaluated in degrees using named preferred and maximum ranges.

For the current single-driver prototype, assume that the largest detected face is the
driver. Additional faces do not by themselves make a frame ineligible, and identity
tracking or ambiguous-driver rejection is not required in this step.

Frame rejection is a technical-quality decision. Passing these checks must not, by
itself, mean that the driver is considered awake or that the frame may be used for
calibration.

The current policy limits are initial engineering defaults. They require validation with
representative camera resolutions, distances, lighting, glasses, face geometry, and the
eventual in-vehicle camera position. This validation may tune frame-quality limits but
must not silently alter fatigue thresholds or unsafe eye-closure duration.

## 1B. Calibration Eligibility / Rolling Awake-State Gating

### Status

Implemented as a deterministic, time-based rolling gate. It consumes Step 1A quality
results and exposes eligibility, current-sample approval, weighted evidence duration,
history duration, and explicit reason codes in monitoring state.

### Goal

Decide whether recent valid driver data is sufficiently consistent with an awake state
before passive personalised calibration is allowed to use it.

### Requirements

Reject or freeze candidate calibration data when there is:

- suspected fatigue
- prolonged eye closure
- unstable or implausible EAR behaviour
- sustained or repeated suspicious mouth-opening behaviour
- a meaningful downward EAR trend
- insufficient recent valid history
- a recent gap in usable face observations
- other low-confidence temporal conditions

Do not decide eligibility from a single EAR threshold alone.

Consider recent temporal behaviour so that a naturally low but stable EAR can be distinguished from decreasing, unstable, or prolonged closure patterns where possible.

Use multiple indicators of awake-consistent behaviour. These may include stable facial
measurements, acceptable recent mouth behaviour, absence of prolonged closure, and
recovery after short blink-like EAR drops. A stable low EAR alone must not be treated as
proof that the driver is awake.

Eligibility must be continuously re-evaluated rather than granted permanently for the
session. When fatigue becomes plausible, immediately stop approving new calibration
samples and discard or quarantine recent samples that may contain fatigue onset.

The rolling gate should tolerate intermittent degraded and rejected frames rather than
requiring a perfect stream. Valid frames should provide the strongest evidence, degraded
frames may provide weaker evidence, and rejected frames must not contribute measurements.
A small number of poor frames should delay eligibility rather than reset all progress.

The current implementation defines and tests:

- a time-based rolling observation window
- the minimum duration and amount of usable recent evidence
- how valid and degraded observations contribute to confidence
- how rejected frames and short face gaps delay eligibility
- which safety-significant events freeze or reset eligibility
- how recent samples are quarantined when fatigue becomes plausible
- temporal EAR stability, trends, prolonged closure, and blink-like recovery
- recent MAR behaviour as corroborating evidence rather than a single-condition decision
- diagnostic state explaining why rolling eligibility has not yet been reached

The initial policy uses a 10-second rolling window, requires at least 8 seconds of
history and 6 weighted seconds of usable evidence, gives degraded frames half weight,
and allows no more than a 1-second gap from the latest awake-consistent measurement.
Valid and degraded samples are credited by elapsed time with a per-observation cap, so
eligibility does not depend on camera frame rate.

An EAR below the existing global `0.3` closure threshold immediately withholds the
current sample. A closure that recovers within the existing `0.5`-second blink grace
preserves history; a longer closure quarantines the preceding 2 seconds and starts a
5-second recovery freeze. Confirmed fatigue and sustained or repeated suspicious mouth
opening use the same quarantine/freeze mechanism. EAR variability and a meaningful
downward linear trend also block eligibility. These are conservative initial engineering
defaults and require validation against representative drivers, cameras, lighting,
glasses, and in-vehicle mounting positions before production use.

### Safety rule

If confidence is insufficient:

- do not calibrate
- do not update a personal baseline
- continue using global fatigue defaults

Never assume the driver is awake merely because the session has just started.

---

## 2. Passive Personalised Calibration

### Status

Deferred until after the local intervention and core runtime-architecture milestones.
Steps 1A and 1B remain active and diagnostic, but no personal profile is currently built
and the global fatigue model remains the safety fallback.

### Goal

Learn driver-specific facial baselines in the background without requiring a dedicated calibration screen or mandatory wait period.

### Initial targets

Learn:

- personal open-eye EAR baseline
- resting-mouth MAR baseline

### Requirements

- collect only samples approved by Step 1
- use robust statistics such as medians and percentiles
- validate sample count, stability, and quality
- keep learned values within safe global bounds
- do not require a guided yawn initially
- do not continuously adapt once fatigue becomes plausible
- fall back to global defaults if calibration never becomes trustworthy

Initial implementation should be session-only.

Saved driver profiles may be explored later.

---

## 3. Hybrid Intervention Controller

### Status

The deterministic local intervention path is implemented. Confirmed fatigue starts a
numbered episode, immediately requests a non-blocking local speaker alarm, keeps the
visual warning active, repeats after a 15-second cooldown while fatigue persists, and
records observed recovery. Missing observations do not falsely close an active episode.

Optional online AI behavior remains deferred to Step 7 and must never gate the initial
alert. The current local warning is an alarm tone rather than generated speech.

### Goal

Turn fatigue confirmation into a real driver-safety response.

### Desired flow

Fatigue confirmed
-> immediate local alarm / visual warning
-> deterministic local spoken warning where appropriate
-> optional online AI voice intervention
-> local fallback if internet or AI is unavailable
-> monitor recovery
-> escalate if fatigue continues

### Requirements

- local alert occurs before any network-dependent call
- initial safety warning must not depend on OpenAI
- introduce fatigue episode IDs
- introduce intervention cooldowns
- avoid firing the same alert on every frame
- keep the first escalation policy simple and deterministic

---

## 4. Refactor Fatigue Scoring Around Calibration

### Status

Deferred with Step 2 because calibrated scoring requires a trustworthy personal profile.

### Goal

Use personalised measurements when a valid calibration profile exists.

### Requirements

- replace hard-coded global EAR/MAR normalization where appropriate with calibrated severity
- retain global fallback behaviour
- bound personal values using validated safe limits
- keep unsafe-duration rules globally controlled initially
- calibration should account for facial geometry, not redefine what counts as unsafe behaviour

---

## 5. Separate Camera / Inference from Flask

### Status

Implemented with one application-owned monitoring worker. The worker opens and retries
camera index `0`, captures and processes each frame once, owns the temporal fatigue,
calibration, intervention, alarm, and benchmark objects, and publishes only the latest
encoded JPEG. HTTP stream clients subscribe to that shared snapshot and no longer own
camera or inference lifecycles.

### Goal

Make the monitoring pipeline continuous and independent from HTTP streaming.

### Desired architecture

Camera source
-> capture worker
-> vision processing
-> fatigue engine
-> monitoring state
   -> intervention controller
   -> Flask/dashboard

### Requirements

- the application owns the camera
- `/video_feed` does not open its own camera pipeline
- capture each frame once
- run inference once per frame
- UI consumers read the latest processed state/frame
- prefer latest-frame behaviour over processing stale frame backlogs
- treat the webcam as the current implementation of a future hardware camera source

---

## 6. Replace Global `data_store` with `MonitoringSession`

### Goal

Make runtime state explicit and maintainable.

A monitoring session should eventually hold:

- calibration state/profile
- current driver metrics
- fatigue state
- current fatigue episode
- intervention state
- assistant state

For the current edge-device architecture, assume one active driver/device session unless requirements change.

Do not introduce distributed multi-user infrastructure such as Redis without a real need.

---

## 7. Make the AI Assistant Non-Blocking and Resilient

### Status

Partially implemented outside the safety-critical path: Flask/camera startup no longer
imports optional speech, LangChain, or OpenAI modules, and assistant import/runtime
failures return an unavailable response without preventing local monitoring. The agent
uses the LangChain 1.x `create_agent` API and structured output rather than the legacy
`AgentExecutor` API.

The assistant request itself remains synchronous and blocking. Background execution,
listening limits, overlap prevention, and deterministic spoken fallback remain future
work in this step.

### Goal

Keep optional AI work separate from the safety-critical monitoring loop.

### Requirements

- vision/fatigue processing must continue while speech/AI work runs
- add listening timeouts
- add phrase duration limits where useful
- prevent overlapping assistant sessions
- handle speech/API failures explicitly
- preserve deterministic local fallback behaviour
- keep assistant state associated with the monitoring session
- favour concise, voice-first responses suitable for driving

---

## 8. Add Stronger Tests and Benchmarking

### Test areas

- EAR/MAR calculations
- normalization
- calibration eligibility
- calibration profile generation
- calibration fallback behaviour
- fatigue state transitions
- intervention cooldowns
- failure handling
- camera/pipeline integration
- relevant Flask routes

### Benchmark separately

1. raw fatigue onset -> fatigue confirmed
2. fatigue confirmed -> local intervention delivered
3. intervention -> driver recovery

### Quality metrics where practical

- false-positive rate
- false-negative rate
- sensitivity / detection rate
- calibration success/failure rate
- processing FPS
- per-frame latency
- intervention latency

Do not count entering a placeholder function as successful intervention delivery.

---

## 9. Add Lightweight Persistence if Useful

### Goal

Retain derived session/event information for evaluation.

Potential records:

- `MonitoringSession`
- `CalibrationProfile`
- `FatigueEpisode`
- `InterventionEvent`

SQLite is sufficient initially.

Prefer storing derived measurements and timestamps rather than raw facial video.

Avoid persisting biometric imagery unless genuinely necessary.

---

## 10. Production / Network Security and Deployment Hardening

Prioritize only when EEPY becomes network-accessible or moves beyond the local prototype.

Potential work:

- authentication
- authorization
- CORS restrictions
- CSRF protection where applicable
- rate limiting
- structured logging
- health/readiness checks
- camera error boundaries
- production server/deployment configuration
- secret management

These should not distract from the core safety functionality during the current prototype phase.

---

# Implementation Order

The roadmap numbers describe the major product improvements, but implementation dependencies may justify completing them in a slightly different sequence.

The original calibration-first sequence was:

1. calibration eligibility / hard frame rejection
2. calibration eligibility / rolling awake-state gating
3. passive personalised calibration
4. refactor fatigue scoring around calibration
5. hybrid intervention controller

Then continue through the architectural and reliability improvements.

Calibration is now intentionally deferred. The active implementation sequence is:

1. hybrid intervention controller with immediate local alert
2. separate camera/inference ownership from Flask
3. replace global `data_store` with `MonitoringSession`
4. make the optional AI assistant non-blocking and resilient
5. strengthen integration tests and benchmarking throughout those changes
6. return to passive personalised calibration
7. refactor fatigue scoring around a validated calibration profile
8. consider persistence and production hardening when justified

This reordering does not remove Steps 1A or 1B. They continue to expose calibration
eligibility diagnostics, but do not affect the global fatigue detector. Step 4 must not
begin before Step 2 produces a trustworthy bounded profile.

Only implement one requested milestone at a time.

---

# Current Focus

Completed steps:

## Step 1A: Calibration Eligibility / Hard Frame Rejection

## Step 1B: Calibration Eligibility / Rolling Awake-State Gating

## Step 3: Hybrid Intervention Controller / Deterministic Local Path

## Step 5: Separate Camera / Inference from Flask

Next step:

## Step 6: Replace Global `data_store` with `MonitoringSession`

Immediate design question:

> How should EEPY replace the mutable global dictionary with one explicit, thread-safe
> session owner for monitoring, calibration, intervention, and assistant state?

The global fatigue detector remains active throughout and acts as the safety fallback.
