# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LibraVision is a Computer Vision and AI project that recognizes Brazilian Sign Language (Libras) alphabet gestures in real-time using a webcam and translates them to text on screen.

**Tech Stack:**
- Hand detection: MediaPipe Hands
- Camera capture: OpenCV
- Data processing: NumPy / Pandas
- Classification model: Scikit-learn (Random Forest)
- Language: Python 3.8+

## Project Structure

The project follows a sequential pipeline architecture with 4 main scripts:

```
LibraVision/
├── data/
│   └── libras_data.csv          # Collected dataset (63 features per sample)
├── models/
│   ├── libras_model.pkl         # Trained Random Forest model
│   └── confusion_matrix.png     # Model evaluation visualization
├── scripts/
│   ├── 1_collect_data.py        # Data collection via webcam
│   ├── 2_train_model.py         # Model training with feature engineering
│   ├── 3_test_model.py          # Model evaluation and metrics
│   └── 4_real_time_app.py       # Real-time inference application
└── requirements.txt
```

## Development Workflow

### Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Sequential Execution Order

The scripts must be run in numerical order as each depends on outputs from previous steps:

1. **Collect Data:**
   ```bash
   python scripts/1_collect_data.py
   ```
   - Collects 40 samples per letter for 27 letters (A-Z + Ç)
   - 10-second preparation time before each letter
   - Saves to `data/libras_data.csv`
   - Press 'q' to quit early

2. **Train Model:**
   ```bash
   python scripts/2_train_model.py
   ```
   - Applies feature engineering (landmark normalization relative to wrist)
   - Trains Random Forest classifier (150 estimators, max_depth=20)
   - 80/20 train-test split with stratification
   - Saves model to `models/libras_model.pkl`

3. **Test Model (Optional):**
   ```bash
   python scripts/3_test_model.py
   ```
   - Generates classification report and confusion matrix
   - Saves visualization to `models/confusion_matrix.png`

4. **Run Real-Time Application:**
   ```bash
   python scripts/4_real_time_app.py
   ```
   - Runs live gesture recognition
   - Press 'q' to exit

## Key Architecture Details

### Feature Engineering Pipeline

The model uses **relative landmark coordinates** instead of absolute positions:

1. Raw input: 21 hand landmarks × 3 coordinates (x, y, z) = 63 features
2. Normalization: All landmarks are made relative to wrist position (landmark 0)
3. This makes the model translation-invariant and improves generalization

**Implementation locations:**
- Training normalization: `scripts/2_train_model.py:23-36`
- Inference normalization: `scripts/4_real_time_app.py:39-42`

### Prediction Smoothing System

The real-time app uses a buffer-based smoothing mechanism to stabilize predictions:

- `PREDICTION_BUFFER_SIZE = 10`: Analyzes last 10 predictions
- `CONFIDENCE_THRESHOLD = 0.8`: Minimum confidence to accept prediction
- Stable prediction requires 70% consensus in buffer (7/10 frames)

**Implementation:** `scripts/4_real_time_app.py:18-21, 52-62`

### MediaPipe Configuration

Hand detection settings optimized for real-time performance:
- Data collection: `min_detection_confidence=0.5, min_tracking_confidence=0.5`
- Real-time app: `min_detection_confidence=0.7, min_tracking_confidence=0.7` (stricter for better accuracy)

## Important Implementation Notes

### Data Consistency

When modifying feature extraction, ensure consistency between training and inference:
- Both `2_train_model.py` and `4_real_time_app.py` must use identical normalization logic
- The wrist (landmark 0) is always the reference point for relative coordinates

### Model Persistence

- Model is saved with `joblib` for fast loading
- The model stores `classes_` attribute used for prediction mapping
- Always use the same scikit-learn version for compatibility

### Path Handling

All scripts use absolute paths resolved from script location:
```python
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
```

This ensures scripts work regardless of where they're executed from.

## Common Issues

### Camera Not Opening
- Check webcam permissions
- Verify camera is not in use by another application
- Try changing camera index in `cv2.VideoCapture(0)` to 1 or 2

### Model Not Found
- Ensure you've run `2_train_model.py` before `3_test_model.py` or `4_real_time_app.py`
- Check that `models/libras_model.pkl` exists

### Poor Recognition Accuracy
- Ensure good lighting conditions
- Keep hand clearly visible in frame
- Train the model with your own hand gestures (each person's hand is different)
- Increase `NUM_SAMPLES` in `1_collect_data.py` for more training data
