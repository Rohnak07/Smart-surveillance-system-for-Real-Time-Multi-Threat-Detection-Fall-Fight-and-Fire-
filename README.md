# Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire)

Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire) analyzes one webcam, IP Webcam phone feed, or video file at a time. It displays and logs camera-based fire, fall, possible medical-emergency, and fight alerts for human review. Alerts are not diagnoses or proof that an event occurred.

> **Credit:** The project idea, evidence/Telegram flow, and violence-CNN checkpoint format are based on [AI-Driven Autonomous Aerial Surveillance](https://github.com/naimurhamim/ai-driven-autonomous-aerial-surveillance) by MD Naimur Hamim, Redwan Bin Jalal, and Md. Ashiqussalehin, under the MIT license (see LICENSE). The project retains a subset of the original fire/violence work, reimplements the fixed-camera logic, and adds fall detection. Drone, accident, and weapon modules were removed.

## Detection methods

| Alert | Current method | Behavior and limits |
|---|---|---|
| **Fire detected** | Fire/smoke YOLO model when models/fire_smoke.pt is present; otherwise an OpenCV heuristic | The heuristic looks for a warm flame core, tapered flame-like shape, and changing flame tip over time. It works on flames visible directly or on a display. Static red/orange objects and bright regions alone are insufficient. The heuristic can still miss unusual, distant, washed-out, or very small flames and cannot reliably distinguish a live flame from a video of one. Smoke requires the trained YOLO model. |
| **Fall detected** | YOLO pose, per-person tracking, body angle/box geometry, and timing | A lying person is reported after the configured short confirmation interval, including someone already lying when first seen. Pose visibility and camera angle affect detection. |
| **Possible medical emergency** | Per-person pose movement, with image-region motion as a fallback around a confirmed fall | No detected movement for fall.medical_still_s triggers an alert and laptop siren. Movement clears that person's emergency state. A missing person is retained briefly to tolerate pose gaps; a changed scene or a confirmed disappearance clears the siren. This rule can alert on someone intentionally staying still, so a person must review it. |
| **Fight detected** | YOLO pose and repeated high-speed wrist/ankle impacts into another person's box | Two impacts from one pair or a connected group of impacts can trigger an alert. Standing close, a single touch, or a hand resting on a shoulder do not count. The current model does not reliably identify forceful grappling or fully occluded fights; only repeated visible strike evidence is confirmed. The optional violence CNN needs a separately trained checkpoint. |

The project currently runs one camera feed per process. Run separate instances with separate configuration files to monitor additional feeds. Pose inference uses yolo11n-pose.pt. CPU speed depends on the laptop, camera resolution, and number/size of people; the default 640-pixel frame width and 416-pixel pose input reduce CPU load but do not guarantee a fixed frame rate.

## Run on Windows

Install the requirements once:

~~~powershell
cd "C:\path\to\Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire)"
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
~~~

Start a laptop webcam:

~~~powershell
python app.py --source 0
~~~

Start IP Webcam on the phone, connect the laptop to the same hotspot, then run:

~~~powershell
python run_phone.py
~~~

If automatic discovery does not find the phone, use its IP shown by IP Webcam:

~~~powershell
python run_phone.py --ip 192.168.43.1 --rotate 90
~~~

The included Start KAVACH AI.cmd launcher starts phone discovery. A fire video on a laptop/phone display can be detected in the standard mode when its flame shape and motion are visible. Press D to enable the more permissive screen-fire test mode, T to test laptop audio, and Q or Esc to quit.

Other sources:

~~~powershell
python app.py --source videos/test.mp4
python app.py --source http://PHONE_IP:8080/video --rotate 90
python app.py --source videos/test.mp4 --no-display --save annotated.mp4
~~~

The application window and every alert identify the configured camera ID and location. Edit camera.id, camera.location, camera.source, and the detector thresholds in config.yaml.

## Alerts and evidence

Events are appended to data/events/events.csv; evidence images are saved under data/evidence/. Optional Telegram alerts require values in .env and alerts.telegram.enabled: true. The repeating medical-emergency siren plays through the computer's default audio output on Windows.

## Fire model

No fire/smoke model weights are included in this project. Until a trained models/fire_smoke.pt exists, fire detection uses the heuristic above. To train a YOLO model on the D-Fire dataset:

~~~powershell
python tools/train_fire.py --epochs 50
copy runs\detect\fire\weights\best.pt models\fire_smoke.pt
~~~

After weights are installed, fire.mode: auto selects the YOLO model. Fire/smoke accuracy must be measured on labeled clips before reporting a score.

## Fight model

The optional violence CNN can provide a second opinion after a suitable checkpoint is trained:

~~~powershell
python tools/split_videos.py violence_raw violence_split
python tools/train_violence_cls.py --data violence_split --epochs 5 --pretrained
~~~

Set violence_cnn.enabled: true and point violence_cnn.model to the trained checkpoint. Until then, the fight alert uses visible repeated pose-based strikes and does not claim to recognize grappling reliably.

## Verification

~~~powershell
pytest -q
python tools/evaluate.py --labels labels.csv --out results
~~~

Evaluation labels use fall, fight, fire, or normal. Do not report accuracy unless it has been measured on labeled clips representative of the target camera setup.
