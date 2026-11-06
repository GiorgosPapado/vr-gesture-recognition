# Data card: VRGestures v2 (draft)

Status: draft. Every number below was computed from the recordings and the author's check file on the
preparation machine with the scripts in this folder (`manifest_summary.md`, `manifest_summary.json`,
`data_stats.json`). Facts only the author knows are marked `TODO(author)` and are intentionally not filled in.
The recordings are not part of this repository.

## Contents (from the data)

- 24 classes: 11 gestures per hand (Click, Close, Grab, Home, Infinity, OpenPalm, Push, Rotate, Point, Scroll,
  Back) for the left and the right hand, plus the two-hand gestures Teleportation and Zoom. Canonical names:
  `Left<Gesture>`, `Right<Gesture>`, `Teleportation`, `Zoom`; ids 0-10 left, 11-21 right, 22 Teleportation, 23 Zoom
  (`classes.py`).
- Recordings root: 58 session folders (`<hash> <Month> <HH> <MM>`) with 7,846 `.json` recordings, plus 26 loose
  `.json` files in the root (7,872 on disk). File names: `<Class> <n> <hash> <DD> <Month> <HH> <MM> <k>.json`.
  No basename occurs twice.
- The author's check file has 7,460 entries (7,219 true, 241 false); 7,425 on-disk files are covered, 447 are not
  (320 + 124 of them are the two whole sessions `af2c212 October 09 43` and `3f8f813 October 10 42`), 35 entries
  point to files that are not on disk.
- Filename date range of the included recordings: September 28 09:55 to October 11 14:57 (the year is not
  encoded in the file names; 2,297 files from September, 4,861 from October).
- File sizes: on disk 15.2 GB in total (7,872 files), median 1.67 MB, mean 1.93 MB, max 12.1 MB; included set 14.0 GB.

## Cleaning rules (author's decisions)

| rule | result |
|---|---|
| check value `false` -> excluded (`flagged_false`) | 241 |
| not in check file -> excluded until verified (`unchecked`) | 447 |
| check entry without a file (`not_found_on_disk`, listed only) | 35 |
| loose files in the root (named sample, one per class plus `LeftOK`/`RightOK`) and folders `AggelosTest`, `IoannaTest` (named after individuals, consent unknown) -> `excluded_folder` | 26 (the two test folders were not present under this root) |
| **included** (true, on disk, not excluded) | **7,158** |

The 26 loose root files are all `true` in the check file; counting them would give 7,184 (= 7,219 true - 35
missing), so the expected 7,184 is reproduced exactly only if those 26 files are included. They are excluded
here because they are the named sample. Included per class and per session: see `manifest_summary.md`
(per hand: right 3,667, left 3,048, both 443; 56 sessions contribute, the two fully unchecked ones do not).

Class-name anomalies: the check file and the root contain 50 distinct leading name tokens instead of 24. The 24
canonical tokens occur in the session files without any variant. The other 26 tokens are the loose root files
(`LeftBack.json` ... `Zoom.json`, 24 of them map to a canonical class) and `LeftOK.json` / `RightOK.json`, which
are not among the 24 classes (unmapped). No case differences or alternative spellings occur in session files; the
mapping table in `classes.py` nevertheless accepts case and separator variants explicitly. For all 7,158
included recordings the `class_name` stored inside the JSON equals the manifest class (0 mismatches).

## Recording format (from the data)

Top-level keys `frameData`, `class_name`, `class_ID`. `frameData` has `left_hand` / `right_hand` (list of frames,
each with 24 joints: `pos` x,y,z and `rotation` quaternion x,y,z,w, hand-local, joint 0 = (0,0,0)),
`left_hand_global` / `right_hand_global` and `hmd` (`pos`, `rotation`, one pose per frame), and `timestamps` (seconds,
one per frame). A hand not used in a gesture is an empty list. In the 300-recording sample: left-hand gestures
have only the left hand (138), right-hand gestures only the right (145), Teleportation/Zoom both (17). `class_ID` in
the files uses a per-hand numbering (e.g. LeftClick 9 vs RightClick 11); use the manifest `class_id`.

Random sample of 300 included recordings (seed 0):

| statistic | min | median | mean | p95 | max |
|---|---|---|---|---|---|
| frames | 56 | 141.5 | 168.3 | 317.7 | 660 |
| duration (s) | 1.2 | 3.0 | 3.2 | 5.3 | 10.0 |
| frame rate (Hz) | 18.3 | 54.8 | 54.5 | 71.6 | 71.9 |

The frame rate is far from constant across recordings (p5 35 Hz); resampling to a fixed length ignores real time.

## Splits and baselines

Participant ids are not in the files. Splits are grouped by session folder (a proxy; the same person may appear in
several sessions, so results may be optimistic for unseen people). Default: 70/15/15 of sessions, seeded, with a
deterministic seed scan so that every class appears in every split. See `splits.csv`, `splits_report.json`,
`BASELINES.md`.

## To be provided by the author

- TODO(author): device / headset used.
- TODO(author): hand-tracking SDK and version (the joint layout has 24 joints per hand).
- TODO(author): capture protocol (environment, posture, repetitions per session, recording trigger).
- TODO(author): number of participants and how sessions map to participants.
- TODO(author): consent / ethics statement.
- TODO(author): instructions given to participants.
- TODO(author): licence decision for v2.
- TODO(author): relation to the Zenodo v1 record (doi 10.5281/zenodo.8027807, a different set: 15 controller + 11 per
  hand + 2 two-handed gestures).
- TODO(author): year of the recordings and meaning of the check-file procedure (who/how recordings were verified).

## From the CGI 2023 paper (about v1; NOT verified for this JSON v2 set)

Source: Papadopoulos, Doumanoglou, Zarpalas, CGI 2023 (LNCS); dataset DOI 10.5281/zenodo.8027807. These facts
describe the published v1 and were supplied by the coordinator; none of them is checked against the v2 files. The v2
`TODO(author)` items above stay open until the author confirms what applies to v2.

- Device Meta Quest 2; Unity3D (left-handed coordinate system). Two Unity recording apps: controller app (SteamVR SDK)
  and hand app (Meta Oculus Integration + Voice SDK: a spoken predefined word starts and another stops recording; a
  voice command can erase a recording).
- Hand skeleton: 24 joints per hand (Oculus SDK); positions and orientation quaternions per joint, per hand as a
  whole and for the HMD, plus per-frame timestamps. One-handed recordings leave the other hand's array empty.
  v1 is stored as .bin files holding numpy arrays. Left-hand gestures mirror right-hand gestures.
- Participants were asked to repeat each gesture slightly differently. Recordings were reviewed with animation tools
  and only those not resembling the target gesture were removed (momentary HMD tracking loss was kept).
- v1 hand set: 34 subjects (25 right-handed, 9 left-handed), 11 one-handed gestures per hand + 2 two-handed;
  averages 307 samples/gesture right, 226 left, 266 both-handed. v1 controller set: 16 subjects (12 right, 4 left),
  15 classes, about 570 samples/class right-handed and about 120 left; anonymous sampling (not matchable to subjects).
  The hand set stored a subject id.
- Participants: computer-science related, age 25-40, little or no previous VR hand-tracking experience; "All
  participants were informed of the procedures and signed an informed consent form". Funding: H2020 EC project
  INFINITY, grant 883293.
- Paper benchmark (sktime Catch22, IndividualTDE, KNeighborsTimeSeries, Rocket; interpolation to 180 timestamps;
  translation invariance by subtracting frame 0; 70/30 split with 5-fold CV): on the 5-gesture right-hand subset
  accuracy Catch22 0.753, Rocket 0.692, IndividualTDE 0.559, KNN 0.368. That subset is Infinity, Scroll, Back, Grab,
  Point; it can be selected from the v2 manifest for a rough comparison, but protocols differ (session-grouped
  split here, different features and classifiers), so numbers are not directly comparable. Not implemented as a
  baseline option yet.

### Subject / participant field in the v2 JSON files

Checked a random sample of 500 included recordings: top-level keys are always `class_ID`, `class_name`, `frameData`;
`frameData` keys always `hmd`, `left_hand`, `left_hand_global`, `right_hand`, `right_hand_global`, `timestamps`; hand
frames only `pos` and `rotation`. No subject / participant field was found (not every file was parsed, but the schema
was uniform in the sample). Participant-wise splits are therefore not possible from the files; the 7-character hash
in the session folder name and the 5-character token in each file name are not documented as subject ids
(TODO(author): say whether the session hash corresponds to a participant).
