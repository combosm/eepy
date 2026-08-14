# Reference diagrams

Landmark maps for the dlib 68-point face predictor
(`models/shape_predictor_68_face_landmarks.dat`). They are here to explain the index
constants in `scripts/camera.py`, which are otherwise unreadable magic numbers.

Nothing in the codebase loads these files — they are documentation for humans.

## The two files

| File | Numbering | Notes |
|---|---|---|
| `facial_landmarks_68_0_indexed.png` | 0–67 | **Matches the code.** Use this one when reading or editing `camera.py`. |
| `facial_landmarks_68_1_indexed.webp` | 1–68 | Matches most published tutorials and the dlib paper. Kept because external references you find online will almost always use this numbering. |

Both show the same landmark layout. They differ only by an offset of one.

> **Read this before changing any index.** `predictor` returns points addressed `0..67`, so
> the code is 0-indexed and only the `.png` applies. The 1-indexed diagram is the more common
> one online, which makes it the easy way to introduce an off-by-one — every constant below
> would look one short if checked against it.

The `.webp` file carried a `.png` extension until it was renamed here; the contents were
always WebP.

## What the constants mean

From `scripts/camera.py`:

```python
LEFT_EYE  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH     = [48, 50, 52, 54, 56, 58]
```

- **`LEFT_EYE` / `RIGHT_EYE`** — the full six-point contour of each eye, in the order dlib
  emits them: outer corner, two upper-lid points, inner corner, two lower-lid points.
- **`MOUTH`** — every *other* point of the twelve-point outer lip contour (48–59). Sampling
  alternate points gives six landmarks arranged like an eye, so the same aspect-ratio maths
  works on both.

Order matters as much as membership. `eye_aspect_ratio()` and `mouth_aspect_ratio()` both
index positionally — `[1]`/`[5]` and `[2]`/`[4]` are the vertical pairs, `[0]`/`[3]` the
horizontal span. Reordering a list silently changes the ratio rather than raising an error.

The inner-lip points (60–67) are unused; mouth opening is measured from the outer contour.
