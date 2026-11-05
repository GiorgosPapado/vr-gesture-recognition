# Controller and Hand Gesture Recognition for Interactions in Virtual Reality Environments

**Goal**
- [ ] Real-time, online controller/hand gesture recognition with good accuracy
- [ ] Good generalization performance with few samples.
  - [ ] Validate: Cross-Validation
  - [ ] Validate: Boot-strapping
  - [ ] Validate: permutation hypothesis test
- [ ] Augmentations: Noise addition & Rotation/Translation transformation of the trajectory
- [ ] Indicate and predict in real-time, which gesture is being performed. Give indicator on the progress of the gesture and signal when gesture is completed
- [ ] Test on augmented (translated/rotated + noise data)

**SkTime Pre-Processing**
- [x] TSInterpolator
- [ ] IntervalSegmenter

**SkTime Benchmark**
- [x] CNNClassifier (2017)
- [x] FCNClassifier (2017) - No sweep
- [x] MLPClassifier (2017) - No sweep
- [x] TapNetClassifier (2020)
- [x] IndividualTDE (2020) - Extension of BOSS
- [x] MUSE
- [x] KNeighborsTimeSeriesClassifier
- [x] Catch22
- [x] SignatureClassifier (2020)
- [x] DrCIF
- [x] Rocket
  
**Proposed**
- [ ] Unsupervised pre-training: Masked-autoencoders (generative model) or Barlow-Twins, 
- [ ] LSTM based, few layers, keras, onnx export

- Related Work:
  - Skeletal Twins: https://ieeexplore.ieee.org/abstract/document/9859595?casa_token=LPsnBrucsKgAAAAA:hPlW1HtvHtvyv6EXweJj8N9Fn84LMT7xU0tcgRZ5VWH2ihUm62fJ0Zw4P8P8kNa20OwUfDovr0g
  - Barlow Twins: https://arxiv.org/pdf/2103.03230.pdf
  - BOSS: ["The BOSS is concerned with time series classification in the presence of noise"](https://link.springer.com/article/10.1007/s10618-014-0377-7)
  - Matthew Middlehurst, James Large, Gavin Cawley and Anthony Bagnall “The Temporal Dictionary Ensemble (TDE) Classifier for Time Series Classification”, in proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, 2020.
  - Schäfer, Patrick, and Ulf Leser. “TEASER: early and accurate time series classification.” Data mining and knowledge discovery 34, no. 5 (2020)

**TODO**
*Dataset*
- [ ] Subject info in controller dataset (tag + excel with recording times)
- [ ] Subject info in hand dataset (scripted from excel with recording times)
- [X] detailed_index_meta for hand dataset
- [X] Repeat to make dataset balanced
*Benchmark*
- [X] Write dataloader for hand gesture dataset
- [X] Write train/val/test split generator for hand dataset (stratify)
- [X] Extend controller dataloader to take into account left/right controller gestures
- [X] Write test script that evaluates classifiers in the test set of the dataset
- [ ] Consider classifier hyper-parameters, pre-processing pipeline and noisy augmented input
- [ ] Make sure classifiers are position/rotation invariant with respect to the starting location of the gesture's trajejectory

**Future Work**
- [ ] Evaluation of right/left controller identification algorithm via VRGestures visualization


# NOTES

Initial experiments achieved high scores in terms of accuracy for the datasets of "5" (controller-5),"10" (controller-10) gestures.
Additionally high accuracy was achieved for the dataset of 4 challenges cases with few samples namely "swipe-right","swipe-left","<",">" (dataset name "controller-swipe").
The features that we used were pure controller position. No Orientation or Time.

This felt counter intuitive so we had to put the classifiers under test in real conditions. We created an ONNX exporter for the Rocket Classifier pipeline
And tested live gesture recognition. Initial findings are given below:

1. Controller-Swipe was really bad in true conditions and really sensitive to the looking orientation of the gesture performer. The first thought was to make the controller trajectory translation invariant by subtracting the coordinates of the first point in the trajectory.
2. After removing the translation offset, we tested on Controller-5 with great success. The classifier rarely made mistakes even at almost any looking orientation. The only gesture that was found to be sensitive to the looking orientation of the performer was "S". This may be attributed to the variations inside the dataset in looking orientation of the participants which made the classifier to be almost rotation invariant. The next step that seems reasonable is to try to make the features truly orientation invariant with the ControllerCoordinateTransform (implemented in preproc)
3. Before applying the ControllerCoordinateTransform, we also evaluated the translation invariant version to the controller-10 dataset. The results we very good.
4. Unfortunately the rotation invarital ControllerCoordinateTransform cannot be exported in to ONNX due to the eigen decomposition which is not available in ONNX. An alternative way would be to augment the dataset with rotated versions of the data
5. Future work: Detect negatives/make rotation invariant through augmentatinos/estimate gesture progress in progress bar


# Timing estimates (normally they depend on hyperparameters but that is for an avarage case)

- catch22: 5 mins
- IndividualTDE: 10 mins
- KNeighborsTimeSeries: 2mins
- MUSE: 8 mins
- rocket: 10 mins
- signature: 10mins

Estimated time for hyper-parameter tunning on controller-10: 1350 mins = (45mins x 30budget) = 22.5h