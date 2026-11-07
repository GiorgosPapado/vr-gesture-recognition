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
- Recordings root: 58 session folders (`<hash> <Month> <HH> <MM>`) with 7,846 `.json` recordings, plus 24 loose
  `.json` files in the root (7,870 on disk). File names: `<Class> <n> <hash> <DD> <Month> <HH> <MM> <k>.json`.
  No basename occurs twice.
- The author's check file has 7,460 entries (7,219 true, 241 false); 7,423 on-disk files are covered, 447 are not
  (320 + 124 of them are the two whole sessions `af2c212 October 09 43` and `3f8f813 October 10 42`), 37 entries
  point to files that are not on disk.
- Date range of the included recordings, recomputed from `manifest.csv` (included rows) using the file-name fields
  `<DD> <Month> <HH> <MM>`: September 28 09:55 to October 11 14:57 (the year is not encoded; 2,297 files from
  September, 4,861 from October; the 24 loose root files carry no date, so 7,158 of the 7,182 included files are
  dated). The session-folder names are not used for dates: in `<hash> <Month> <HH> <MM>` the two numbers are an
  hour and minute (e.g. a folder `... October 16 42` holds files dated `07 October 16 57`), not a day, so folder
  names such as "October 12/15/16" do not contradict the range. The month in every folder name equals the month in
  its file names (0 mismatches).
- File sizes: on disk 15.2 GB in total (7,870 files), median 1.67 MB, mean 1.93 MB, max 12.1 MB; exported set (7,180 files) 14.1 GB.

## Cleaning rules (author's decisions)

| rule | result |
|---|---|
| check value `false` -> excluded (`flagged_false`) | 241 |
| not in check file -> excluded until verified (`unchecked`) | 447 |
| check entry without a file (`not_found_on_disk`, listed only) | 37 |
| folders `AggelosTest`, `IoannaTest` (named after individuals, consent unknown) -> `excluded_folder` | 0 (not present under this root; rule kept) |
| **included in the manifest** (true, on disk, not excluded; includes the 24 loose root files) | **7,182** |
| of which in the 24-class vocabulary (exported to npz, splits, baselines) | **7,180** |

Author's decision: the 24 loose `.json` files in the root (all `true` in the check file) are part of the dataset;
they form the pseudo-session `(root)`. 7,182 = 7,219 true - 37 missing, as expected. The 24 loose files are
`LeftBack.json` ... `RightScroll.json` (22, mapped to the canonical classes; there is no loose `Teleportation.json` / `Zoom.json` any more) plus `LeftOK.json` and `RightOK.json`:
**included in the manifest, outside the 24-class vocabulary**. Author statement (not independently verified):
`LeftOK.json` / `RightOK.json` are thumbs-up gestures (left hand / right hand). (`class_in_vocabulary`
= False, `class_id` -1; their `class_name` inside the file is `GestureAnimation`, which does not match the intended meaning; this inconsistency is kept as documented; each has one hand, 158 and 135
frames). They are not in the npz, splits or baselines.

Two former loose files, `Teleportation.json` and `Zoom.json`, contained `class_name` "LeftInfinity", so their content did
not agree with their file names. Author statement (not independently verified): they were mislabelled and the author
removed them from the source folders. They are no longer on disk; the check file still lists them, so they appear
as `not_found_on_disk`. All numbers in this card were regenerated after their removal.

Included in the vocabulary per hand: right 3,678, left 3,059, both 443; 57 groups contribute (56 session folders +
`(root)`; the two fully unchecked sessions do not). Per class and per session: see `manifest_summary.md`.

Class-name anomalies: the check file and the root contain 50 distinct leading name tokens instead of 24. The 24
canonical tokens occur in the session files without any variant. The other 26 tokens are the loose root file names in the check file
(24 map to a canonical class from the `X.json` form, of which 22 are on disk; `LeftOK.json` / `RightOK.json` are unmapped). No case
differences or alternative spellings occur in session files; the mapping table in `classes.py` nevertheless
accepts case and separator variants explicitly. For all included in-vocabulary recordings the `class_name` stored inside the JSON equals the manifest class (7,180 of 7,180 match, checked by reading every included in-vocabulary file).

## Recording format (from the data)

Top-level keys `frameData`, `class_name`, `class_ID`. `frameData` has `left_hand` / `right_hand` (list of frames,
each with 24 joints: `pos` x,y,z and `rotation` quaternion x,y,z,w, hand-local, joint 0 = (0,0,0)),
`left_hand_global` / `right_hand_global` and `hmd` (`pos`, `rotation`, one pose per frame), and `timestamps` (seconds,
one per frame). A hand not used in a gesture is an empty list. In the 300-recording sample: left-hand gestures
have only the left hand (131), right-hand gestures only the right (155), Teleportation/Zoom both (14). `class_ID` in
the files uses a per-hand numbering (e.g. LeftClick 9 vs RightClick 11); use the manifest `class_id`.

Random sample of 300 in-vocabulary included recordings (seed 0):

| statistic | min | median | mean | p95 | max |
|---|---|---|---|---|---|
| frames | 61 | 150.0 | 176.8 | 336.1 | 551 |
| duration (s) | 1.0 | 3.1 | 3.2 | 5.7 | 8.0 |
| frame rate (Hz) | 24.0 | 63.7 | 56.1 | 71.6 | 71.8 |

The frame rate is far from constant across recordings (p5 36 Hz); resampling to a fixed length ignores real time.

## Splits and baselines

Participant ids are not in the files, and no session-to-participant mapping is available: the author states that if
it is not in the source folders it no longer exists, and a read-only search of the source folders found none.
Splits are grouped by session folder, so participant independence between splits cannot be guaranteed (the same
person may appear in several sessions, so results may be optimistic for unseen people). The loose root files form
one pseudo-session `(root)`. Default: 70/15/15 of sessions, seeded, with a
deterministic seed scan so that every class appears in every split. See `splits.csv`, `splits_report.json`,
`BASELINES.md`.

Baseline caveats (all numbers in `BASELINES.md` / `baselines_results.json`, regenerated on this manifest and
`splits.csv`): the whole `(root)` pseudo-session lies in the test split (the npz holds 22 of its 24 in-vocabulary
files, without `Teleportation.json` and `Zoom.json`), so test metrics are reported with and without it; test
accuracy is given with a session-cluster bootstrap CI over the 9 test sessions, which is wide; the absent hand is
zero-filled, so left/right/both is readable from the input and a hand-presence-only baseline is reported next to the
11-way within-hand accuracy; the hand-local joint positions are nearly constant in this export (see the ablation in
`BASELINES.md`).

## To be provided by the author

- TODO(author): device / headset used.
- TODO(author): hand-tracking SDK and version (the joint layout has 24 joints per hand).
- TODO(author): capture protocol (environment, posture, repetitions per session, recording trigger).
- TODO(author): number of participants (the session-to-participant mapping is not available, see Splits).
- Consent: author statement (not independently verified): participants gave consent for their gestures to be used.
  TODO(author): exact consent/ethics wording and any approval ids.
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
(no mapping from session hash to participant is available; see Splits).
