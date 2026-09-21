# HandROM

HandROM is a complete Streamlit MVP that estimates active MCP, PIP, and DIP motion for one selected index, middle, ring, or little finger per session from side-view maximum-extension and maximum-flexion photographs. It aggregates repeated views with the median and calculates Total Active Motion (TAM) for later comparison with manual-goniometer measurements.

> Experimental range-of-motion estimate based on uploaded photographs. This application is not a medical diagnosis and is not a replacement for assessment by a qualified hand therapist, surgeon, or clinician using a manual goniometer.

HandROM is **not clinically validated** and makes no accuracy, diagnosis, treatment, normal/abnormal, or medical-grade claims.

## Features

- Three-step clinician workflow: Welcome, Upload, Results and Export.
- Streamlined start and upload screens with an automatically generated internal anonymous case ID.
- Automatic left/right hand detection from normal, unmirrored photographs; no healthy-hand upload or contralateral comparison.
- A one-finger-at-a-time selector with one to three side-view extension photographs and one to three side-view flexion photographs for the selected finger.
- A responsive avatar-hand pose guide labeled for the selected finger, with separate extension and flexion images for all four fingers.
- Safe in-memory JPG/JPEG/PNG validation, EXIF correction, metadata removal, and aspect-ratio-preserving inference resize.
- MediaPipe Tasks Hand Landmarker in `IMAGE` mode with `num_hands=2`, 21 normalized landmarks, 21 world landmarks, and cross-photo handedness consistency validation.
- Target-finger 2D image-plane MCP/PIP/DIP angles with pixel aspect-ratio correction.
- Image quality, target-chain framing, resolution, blur, brightness, handedness, geometry, and per-finger repeatability checks.
- Always-visible annotated photo review with MCP, PIP, and DIP values repeated beneath every image.
- Correct TAM calculation for index, middle, ring, and little fingers.
- A compact TAM summary and a ReportLab PDF report containing the annotated images.
- Optional synthetic demo mode that never resembles patient data.

Not implemented by design: live camera/video, thumb TAM, passive ROM, hyperextension, healthy-hand comparison, TAM percentage, reference classifications, diagnosis, treatment advice, accounts, authentication, database, cloud storage, or external vision APIs.

## Architecture

`app.py` contains Streamlit composition only. The `handrom/` package separates concerns:

- `config.py`, `data_models.py`: constants and typed session/result models.
- `image_processing.py`: secure upload decoding and inference resize.
- `hand_detector.py`: MediaPipe Tasks adapter and exactly-one-hand enforcement.
- `landmark_mapping.py`: stable finger/joint landmark indices.
- `geometry.py`, `palm_coordinate_system.py`, `angle_estimator.py`: validated geometry and side-view joint estimates; legacy 3D helpers remain testable in isolation.
- `quality.py`: image, detection, framing, and repeatability checks.
- `aggregation.py`: visible median/mean/std/min/max/count calculations.
- `tam_calculator.py`: TAM validation and calculation.
- `visualization.py`: non-destructive annotated PNG rendering.
- `analysis_service.py`: one-image pipeline orchestration.
- `exporters.py`, `report_generator.py`: CSV, JSON, and PDF outputs.
- `session_manager.py`: session initialization and privacy-preserving reset.
- `demo_data.py`: synthetic measurement fixtures.

## Technology and Python version

The pinned stack uses Python 3.12, Streamlit, MediaPipe Tasks Vision, NumPy, OpenCV Headless, Pillow, Pandas, ReportLab, Pytest, and Ruff. MediaPipe 1.0.1 advertises Python 3.12 support. Use a 64-bit Python build on Windows/Linux/macOS supported by the pinned packages.

## Installation

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

macOS/Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

For a production deployment, install `requirements.txt` instead. It contains only runtime dependencies.

### MediaPipe model setup

Download the official model bundle to the exact required path:

```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task', 'models/hand_landmarker.task')"
```

HandROM checks for a missing model and reports a setup error instead of crashing. The detector creates `HandLandmarkerOptions` with `RunningMode.IMAGE`, `num_hands=2`, and detection/presence/tracking thresholds of 0.7. Final target-finger angles use normalized image landmarks scaled back to pixel coordinates; MediaPipe world-landmark depth is not used for the one-finger side-view measurements.

### Pose-guide assets

Blue avatar-hand pose guides are bundled for the active workflow. Each finger displays separate transparent extension and flexion images side by side. The Middle guides use an upper, back-of-hand view, while the Ring and Little guides use an under, palm-facing view. Only the eight runtime assets are included in the deployment package:

```text
assets/pose_examples/index_extension_avatar_hand.png
assets/pose_examples/index_flexion_avatar_hand.png
assets/pose_examples/middle_extension_avatar_hand.png
assets/pose_examples/middle_flexion_avatar_hand.png
assets/pose_examples/ring_extension_avatar_hand.png
assets/pose_examples/ring_flexion_avatar_hand.png
assets/pose_examples/little_extension_avatar_hand.png
assets/pose_examples/little_flexion_avatar_hand.png
```

These are illustrative pose aids, not clinician-approved photographs. Follow the app's separate side-view capture checklist for camera placement. See `assets/pose_examples/SOURCES.md` for provenance and replacement requirements.

## Run and developer checks

```bash
streamlit run app.py
pytest
ruff check .
```

## Deploy

### Streamlit Community Cloud

1. Upload this repository to GitHub.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py` as the entry point.
3. Choose Python 3.12 in the app's advanced settings.
4. Deploy. No secrets or environment variables are required for normal operation.

The platform installs Python packages from `requirements.txt` and Debian packages from `packages.txt`. The bundled model and pose assets must remain in `models/` and `assets/pose_examples/`.

### Docker

Build and start HandROM:

```bash
docker compose up --build
```

If Docker exposes Compose as the standalone command on your machine, use `docker-compose up --build` instead.

Open `http://localhost:8501`, then stop the service with:

```bash
docker compose down
```

The container includes the bundled MediaPipe model and pose-guide assets, runs as a non-root user, and binds port 8501 to `127.0.0.1` so the application is available only from the local computer.

The Docker image installs MediaPipe's required Linux EGL/OpenGL ES runtime libraries. For platforms that install Python dependencies directly instead of using the Dockerfile, the root-level `packages.txt` supplies the same Debian packages. After changing either file, rebuild or redeploy the application so the system packages are installed.

Synthetic demo mode:

```powershell
$env:HANDROM_DEMO = "1"
streamlit run app.py
```

```bash
HANDROM_DEMO=1 streamlit run app.py
```

The app labels this mode **“Demo data — not calculated from patient images.”** Demo measurements and placeholder previews are synthetic fixtures, not patient photographs.

## Required photographs

Select one finger for the current session, then provide 1–3 extension and 1–3 flexion photographs for that finger only. Two or three per pose are recommended to assess repeatability. Start another session to measure another finger.

Support the forearm while leaving the hand free to move. Keep the wrist neutral. Place the camera perpendicular to the selected finger's bending plane, approximately 40–70 cm away, with the hand occupying about 70% of the image. Keep the wrist, MCP, PIP, DIP, and fingertip visible against a plain contrasting background with bright, even lighting. Keep the same camera position for extension and flexion. Do not photograph directly above the back of the hand or point the target finger toward the camera.

For maximum active extension, straighten the selected finger without pressing it against a table; move the other fingers away or gently flex them. For maximum active flexion, bend the selected finger fully while keeping the thumb and other fingers away. If the fingertip is hidden, rotate the camera or hand only 10–15° rather than reducing flexion.

For a right hand, photograph index and middle mainly from the thumb side and ring and little mainly from the little-finger side. Reverse the physical hand orientation for a left hand, not the calculation logic. Use normal, unmirrored rear-camera photographs because physical handedness cannot be recovered reliably from a mirrored image without that capture context. Reject or replace photographs with overlap, hidden target joints, cropped wrist/tip, visible wrist flexion/extension, strong foreshortening, target landmarks outside the visible finger, or substantially different extension/flexion camera angles. Because MediaPipe does not expose dependable target-joint visibility, the app provides a target-only landmark overlay for operator review.

## Image validation and privacy

Uploads must be static JPG/JPEG/PNG, no larger than 15 MB, no more than 25 megapixels, and at least 640×480. HEIC, PDF, GIF, animation, video, malformed/truncated data, unexpected MIME types, and misleading extensions are rejected. Pillow corrects EXIF orientation, converts to RGB, creates a new metadata-free array, and preserves aspect ratio during inference resize. Uploaded filenames are used only as display labels and never as storage paths.

Images and patient-specific results remain in the current Streamlit session and are never globally cached, written to permanent storage, uploaded to another service, used for analytics, or used for training. When run locally, processing stays on the local machine. When deployed remotely, the browser necessarily sends uploads to the server hosting Streamlit; the application does not permanently store them. Clear Session removes uploads, landmarks, annotations, and measurements from session state, creates a new UUID, and returns to Welcome.

## Joint-angle method

MediaPipe returns 21 normalized image landmarks and 21 estimated 3D world landmarks. In the active one-finger workflow, HandROM measures the selected finger in the side-view image plane and does not use monocular world-landmark depth.

Normalized x/y coordinates are multiplied by the inference image width and height before angle calculation so non-square images do not distort geometry. For a joint with proximal point `A`, joint `B`, and distal point `C`, HandROM calculates the angle between `B - A` and `C - B`. Straight is approximately 0°, a right angle is approximately 90°, and flexion can approach 180°. It rejects non-finite coordinates, nearly zero bone vectors, invalid cosine values, and out-of-range angles.

PIP triplets are MCP→PIP→DIP; DIP triplets are PIP→DIP→tip for each supported finger.

MCP flexion uses wrist→target MCP→target PIP, with wrist→MCP serving as the visible metacarpal-axis proxy. PIP uses MCP→PIP→DIP, and DIP uses PIP→DIP→tip. This remains an estimate from surface landmarks, not a direct bone-axis measurement.

In maximum extension, remaining MCP/PIP/DIP flexion is the extension deficit; the pose is not calibrated to zero. Deficits below the 3° configured noise floor are reported as 0°. Hyperextension is not measured.

## Aggregation, quality, and TAM

Every included valid image remains visible. HandROM calculates the median, mean, population standard deviation, minimum, maximum, and count separately for each target finger and joint, then uses the median as the final measurement. It does not mix measurements from other fingers, select the single largest flexion value, silently remove outliers, or hide inconsistent images.

Repeatability quality:

- High: at least two valid included images, major checks pass, maximum joint standard deviation ≤5°.
- Medium: one valid image, or maximum standard deviation >5° and ≤10°.
- Low: no valid image, missing required measurement, major detection/geometry failure, included Low-quality image, or maximum standard deviation >10°.

A single image cannot receive High repeatability quality. Brightness is a warning. Blur uses configurable Laplacian variance. HandROM does not claim reliable occlusion detection because MediaPipe does not expose dependable per-landmark visibility for this task. No TAM is produced when extension or flexion measurement quality is Low.

For each finger:

```text
total flexion = MCP maximum flexion + PIP maximum flexion + DIP maximum flexion
total extension deficit = MCP deficit + PIP deficit + DIP deficit
TAM = total flexion - total extension deficit
```

The subtraction is never reversed. Full precision is retained internally and displayed to one decimal place. Example: maximum flexion `90 + 85 + 55 = 230°`; extension deficit `0 + 10 + 5 = 15°`; TAM `230 - 15 = 215°`. Missing, non-finite, out-of-range, and negative results are rejected. There is intentionally no TAM percentage or healthy-hand classification.

## Report and validation data

The results-page footer exposes only the PDF report. The lower-level exporter utilities remain available for research and manual-goniometer validation workflows:

- Automatic-measurement CSV: one row per session, image, pose, finger, and joint with stable IDs, MediaPipe angle, inclusion, quality, confidence, and warnings.
- TAM-summary CSV: final deficits, maximum flexion, totals, TAM, counts, and quality per finger.
- Manual-validation template CSV: matching identifiers and MediaPipe angles with blank manual and error fields.
- JSON: session metadata, per-image measurements/quality/inclusion, aggregate statistics, TAM, and warning codes; no raw images.
- PDF: session details, per-finger values, quality, warnings, method, limitations, disclaimer, and the annotated measurement images.

Do not invent manual data. After collecting paired measurements:

```text
signed_error_deg = mediapipe_angle_deg - manual_goniometer_angle_deg
absolute_error_deg = abs(signed_error_deg)
tam_error_deg = mediapipe_tam_deg - manual_goniometer_tam_deg
```

Stable lowercase finger/joint labels allow joins in Python, Pandas, Excel, or other tools.

## Validation plan

HandROM is not clinically validated. The recommended next step is a preregistered comparison against manual-goniometer measurements across multiple participants, healthy and injured hands (analyzed as validation cohorts, not compared inside this MVP), different hand sizes, skin tones, devices, resolutions, lighting, backgrounds, camera distances, repeated photographs, and multiple examiners. Validate MCP, PIP, DIP, and TAM using mean signed error, mean absolute error, root mean squared error, intraclass correlation where appropriate, Bland–Altman analysis, and test-retest reliability. Do not make accuracy claims until this work is completed.

## Known limitations

- MediaPipe predicts surface landmarks rather than directly measuring bones.
- Side-view accuracy depends on keeping the camera perpendicular to the selected finger's bending plane.
- Wrist→MCP is only a visible proxy for the selected metacarpal axis, especially for ring and little fingers.
- Camera perspective, roll, distance, and extension/flexion viewpoint mismatch can affect measurements.
- Full flexion can cause finger overlap; hidden landmarks may be inferred incorrectly.
- Severe deformity can reduce detection or geometry reliability.
- Overlap, wrist neutrality, and landmark-on-skin placement cannot be established reliably from Hand Landmarker output alone.
- One image cannot measure repeatability.
- Hyperextension, thumb TAM, and passive ROM are unsupported.
- Results are not clinically validated and the app does not provide a diagnosis.

## Future improvements

After formal validation: clinician-approved pose assets; guided capture overlays; calibrated multi-view reconstruction; better overlap warnings; device/camera calibration; validation dashboards for paired manual data; accessible localization; and deployment-specific privacy/security review. These should not be presented as clinical improvements until evaluated.

## References

- [Official MediaPipe Hand Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MediaPipe Hand Landmarker overview and model](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/index)
- [Hand Surgery Resource: Total Active ROM](https://www.handsurgeryresource.net/total-active-rom) — its instruction text appears to reverse the subtraction order; HandROM intentionally uses the corrected formula specified above.
