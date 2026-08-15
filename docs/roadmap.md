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

## 1. Calibration Eligibility / Awake-State Gating

### Goal

Decide whether current driver data is trustworthy enough to learn from before passive personalised calibration is allowed to use it.

### Requirements

Reject candidate calibration data when there is:

- suspected fatigue
- prolonged eye closure
- unstable or implausible EAR behaviour
- poor landmark quality
- unsuitable head pose where detectable
- no usable face
- ambiguous or multiple likely driver faces
- other low-confidence conditions

Do not decide eligibility from a single EAR threshold alone.

Consider recent temporal behaviour so that a naturally low but stable EAR can be distinguished from decreasing, unstable, or prolonged closure patterns where possible.

### Safety rule

If confidence is insufficient:

- do not calibrate
- do not update a personal baseline
- continue using global fatigue defaults

Never assume the driver is awake merely because the session has just started.

---

## 2. Passive Personalised Calibration

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

The immediate sequence is:

1. calibration eligibility / awake-state gating
2. passive personalised calibration
3. refactor fatigue scoring around calibration
4. hybrid intervention controller

Then continue through the architectural and reliability improvements.

Only implement one requested milestone at a time.

---

# Current Focus

Current step:

## Step 1: Calibration Eligibility / Awake-State Gating

Immediate design question:

> How can EEPY determine that observed facial behaviour is sufficiently consistent with an awake driver to safely use those measurements for passive calibration, without assuming that startup measurements represent the driver's normal awake state?

The global fatigue detector remains active throughout and acts as the safety fallback.
