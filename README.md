# Business Entity Resolution Challenge
## Amazon ML Challenge - 72 Hour Hackathon

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-green)

**Matching business records across 3 independent data sources using machine learning and string similarity techniques.**

---

## 📋 Problem Statement

In large-scale commercial platforms, business identity data arrives from multiple independent sources — each contributing partial, noisy fragments of information about the same real-world entities. This project builds an ML solution to determine which records across 3 sources refer to the same real-world business entity.

### Key Challenges:
- **Name variations**: Abbreviations (Corp vs. Corporation), typos, transliterations, word reordering
- **Address noise**: Partial addresses, format variations, missing components, landmark-based references
- **Multiple sources**: No common identifiers across Source 1 (reference), Source 2, and Source 3
- **Open-set countries**: Training has US & India; test includes France (unseen in training)

### Evaluation Metric: **F₀.₅ Score** (Precision-Heavy)
```
F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```
- Penalizes false merges (false positives) 2× more than missed matches
- Macro-averaged across all Source 1 entities
- Correctly predicting singletons (no match) = 1.0 score

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.9+**
- **pip** or **conda**
- ~2-4 GB disk space for data & models
- Training data access

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd business_entity_resolution

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Full Pipeline (End-to-End)

```bash
# From the project root directory
python -m src.pipeline.main \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output/ \
    --validation-split 0.2 \
    --verbose

# Output files generated:
# - output/matching_results.tsv (final matches for leaderboard)
# - output/candidate_pairs.tsv (candidate set for analysis)
# - output/validation_metrics.json (F_0.5 score on validation split)
```

---

## 📁 Project Structure

```
business_entity_resolution/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies (pinned versions)
├── pyproject.toml                     # Project metadata
│
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv         # Source 1 training records (reference)
│   │   ├── train_source2.tsv         # Source 2 training records
│   │   ├── train_source3.tsv         # Source 3 training records
│   │   └── train_ground_truth.tsv    # Ground truth matches
│   │
│   └── test/
│       ├── test_source1.tsv          # Source 1 test records
│       ├── test_source2.tsv          # Source 2 test records
│       └── test_source3.tsv          # Source 3 test records
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                 # Data loading & parsing (Member 1)
│   │   └── preprocessor.py           # Data cleaning & normalization (Member 1)
│   │
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── blocker.py                # Multi-stage blocking strategy (Member 2)
│   │   ├── similarity.py             # String similarity metrics
│   │   └── filters.py                # Blocking filters & thresholds
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── feature_generator.py      # Feature engineering (Member 3)
│   │   ├── string_features.py        # String similarity features
│   │   └── address_features.py       # Address-specific features
│   │
│   ├── model/
│   │   ├── __init__.py
│   │   ├── trainer.py                # Model training & validation (Member 3)
│   │   ├── predictor.py              # Inference & threshold tuning
│   │   └── models.py                 # Model definitions (XGBoost, LightGBM, etc.)
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── main.py                   # End-to-end orchestration (Member 4)
│   │   └── config.py                 # Pipeline configuration
│   │
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── evaluator.py              # F_0.5 score calculation (Member 4)
│   │   └── format_checker.py         # Output format validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                 # Logging utilities
│       └── constants.py              # Global constants
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis (Member 1)
│   ├── 02_blocking_analysis.ipynb    # Blocking strategy experiments (Member 2)
│   ├── 03_feature_analysis.ipynb     # Feature importance & engineering (Member 3)
│   └── 04_model_tuning.ipynb         # Hyperparameter tuning (Member 3)
│
├── output/
│   ├── matching_results.tsv          # Final predictions (SUBMITTED TO LEADERBOARD)
│   ├── candidate_pairs.tsv           # Candidate set before final ranking
│   └── validation_metrics.json       # Performance metrics on validation split
│
├── models/
│   ├── blocker_config.json           # Blocking parameters
│   ├── feature_scaler.pkl            # Feature normalization (if applicable)
│   └── final_model.pkl               # Trained ML model (XGBoost/LightGBM)
│
└── Documentation_template.md         # Methodology write-up (REQUIRED FOR SUBMISSION)
```

---

## 🔧 Detailed Usage

### Step 1: Data Loading & Preprocessing

```bash
python -m src.data.loader \
    --train-source1 dataset/train/train_source1.tsv \
    --train-source2 dataset/train/train_source2.tsv \
    --train-source3 dataset/train/train_source3.tsv \
    --output cleaned_data.pkl
```

**Output**: Preprocessed DataFrames with normalized names, addresses, and basic cleaning.

### Step 2: Blocking / Candidate Generation

```bash
python -m src.blocking.blocker \
    --source1 dataset/train/train_source1.tsv \
    --source2 dataset/train/train_source2.tsv \
    --source3 dataset/train/train_source3.tsv \
    --output candidates.pkl \
    --strategy multi_stage \
    --jaccard-threshold 0.5 \
    --cosine-threshold 0.3
```

**Output**: Candidate pairs for each Source 1 entity.

**Key Metrics**:
- Candidate set size (average candidates per S1 entity)
- Blocking recall (% of ground truth matches in candidate set)
- Reduction ratio (O(n²) → candidate set size)

### Step 3: Feature Engineering

```bash
python -m src.features.feature_generator \
    --candidates candidates.pkl \
    --source1-data dataset/train/train_source1.tsv \
    --source2-data dataset/train/train_source2.tsv \
    --source3-data dataset/train/train_source3.tsv \
    --output features.pkl \
    --feature-set all  # Options: 'all', 'string_only', 'minimal'
```

**Features Generated**:
- **String Similarity**: Levenshtein, Jaro-Winkler, Jaccard, TF-IDF cosine
- **Token Features**: Token overlap, acronym matching, order differences
- **Address Features**: Country match, component matching, format consistency
- **Domain Features**: Name length ratio, special character handling

### Step 4: Model Training & Validation

```bash
python -m src.model.trainer \
    --features features.pkl \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --model-type xgboost \
    --validation-split 0.2 \
    --output models/final_model.pkl \
    --f-beta 0.5
```

**Output**: 
- Trained model (XGBoost / LightGBM)
- Feature importance rankings
- F₀.₅ score on validation split
- Optimal decision threshold

### Step 5: Generate Final Predictions

```bash
python -m src.model.predictor \
    --model models/final_model.pkl \
    --candidates-test output/test_candidates.pkl \
    --threshold 0.5 \
    --output output/matching_results.tsv
```

**Output**: `matching_results.tsv` ready for leaderboard submission.

---

## 📊 Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      INPUT: 3 TSV FILES                          │
│            (train_source1.tsv, train_source2.tsv, ...)           │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   Data Loading   │
                    │   & Cleaning     │ (Member 1)
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │    STAGE 1: BLOCKING │
                    │                      │
                    │ • Fuzzy name match   │
                    │ • Phonetic matching  │
                    │ • Address-based      │
                    │ • TF-IDF similarity  │
                    │                      │ (Member 2)
                    │ Output: Candidates   │
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ STAGE 2: FEATURES    │
                    │                      │
                    │ • String metrics     │
                    │ • Token features     │
                    │ • Address features   │
                    │                      │ (Member 3)
                    │ Output: Feature vec. │
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ STAGE 3: ML MODEL    │
                    │                      │
                    │ • XGBoost/LightGBM   │
                    │ • Binary classifier  │
                    │ • Threshold tuning   │ (Member 3)
                    │   (F_0.5 optimized)  │
                    │                      │
                    │ Output: Predictions  │
                    └────────┬─────────────┘
                             │
                             ▼
                    ┌──────────────────────┐
                    │ STAGE 4: VALIDATION  │
                    │                      │
                    │ • Format checking    │
                    │ • ID validation      │
                    │ • Metrics reporting  │ (Member 4)
                    │                      │
                    │ Output: TSV files    │
                    └────────┬─────────────┘
                             │
                             ▼
┌────────────────────────────────────────────────────────────────┐
│              OUTPUT: 2 TSV FILES (Ready for Upload)             │
│      • matching_results.tsv (LEADERBOARD SUBMISSION)            │
│      • candidate_pairs.tsv (ANALYSIS & DEBUGGING)               │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Key Implementation Details

### Blocking Strategy (Member 2)

**Multi-Stage Blocking** (designed for high recall):

1. **Stage 1 - Token Prefix Matching**
   - Extract first 3-5 tokens from business_name
   - Jaccard similarity ≥ 0.6
   
2. **Stage 2 - Address-Based**
   - Country exact match
   - City/locality extraction and fuzzy match
   
3. **Stage 3 - Phonetic Matching**
   - Soundex/Metaphone on business names
   - Catch typos and transliterations
   
4. **Stage 4 - TF-IDF Cosine**
   - Full name + address TF-IDF vectors
   - Cosine similarity ≥ 0.3

**Combining Stages**: Union of all candidates (high recall, lower precision acceptable at this stage).

### Feature Engineering (Member 3)

**String Similarity Features** (8 features):
- Levenshtein distance (normalized)
- Jaro-Winkler similarity
- Jaccard token overlap
- TF-IDF cosine similarity
- Token overlap count
- Token order difference
- Length ratio (name lengths)
- Common prefix length

**Address Features** (4 features):
- Country exact match (binary)
- Address component overlap (fuzzy)
- Format consistency score
- Missing component penalty

**Total Features**: ~15-20 features for each candidate pair.

### Model Training (Member 3)

**Model Options** (pick best):
- **XGBoost**: Fast training, good F₀.₅ performance, interpretable feature importance
- **LightGBM**: Faster for large datasets, good accuracy
- **Logistic Regression**: Baseline, simple threshold tuning

**Hyperparameter Search**:
```python
params_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
```

**Threshold Tuning** (Critical for F₀.₅):
```python
# Test thresholds from 0.3 to 0.7
best_threshold = find_optimal_threshold(
    val_predictions, 
    val_labels, 
    metric='f_0.5',
    thresholds=np.arange(0.3, 0.75, 0.05)
)
# Typically: 0.5-0.6 (conservative on false positives)
```

### Output Validation (Member 4)

**Run before submission**:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

**Validation Checks**:
- ✅ All Source 1 test entities present
- ✅ No duplicate IDs in any list
- ✅ No self-matches to Source 1
- ✅ All matched IDs exist in test set
- ✅ Tab-separated format
- ✅ Matched IDs ⊆ Candidate IDs (pipeline consistency)

---

## 📈 Expected Performance

### Baseline (Blocking + Simple Threshold)
- **Precision**: ~0.65-0.75
- **Recall**: ~0.70-0.80
- **F₀.₅**: ~0.68-0.76

### Optimized (Tuned Model + Threshold)
- **Precision**: ~0.75-0.85
- **Recall**: ~0.65-0.75
- **F₀.₅**: ~0.75-0.85

**Note**: F₀.₅ is precision-heavy; focus on minimizing false positives.

---

## ⚠️ Critical Constraints & Important Notes

### ❌ STRICTLY PROHIBITED
- ❌ External entity resolution APIs or services
- ❌ Government database lookups
- ❌ Geocoding APIs for address normalization
- ❌ Any external data augmentation
- ❌ Hard-coding country filters (France is in test set!)

### ✅ MUST-DO's
- ✅ Every Source 1 entity in test set **must** appear in output
- ✅ Output format must be **exactly** as specified (tab-separated, no quoting)
- ✅ Validate locally before leaderboard submission
- ✅ Pin all dependencies in `requirements.txt`
- ✅ Model ≤ 8 Billion parameters
- ✅ Use MIT/Apache 2.0 licensed models only

### 🎯 Edge Cases to Handle
- **Singletons**: Entities with no true matches must predict empty list → scores 1.0 if correct
- **France Data**: Test includes unknown country; don't hardcode countries
- **Multiple Matches**: One S1 entity can match many S2/S3 records
- **Partial Addresses**: Some addresses missing components (PIN, state); use what's available

---

## 🛠️ Troubleshooting

### Issue: "ValueError: sep not recognized as a valid delimiter"
**Solution**: Always load TSV files with `sep="\t"`:
```python
df = pd.read_csv("file.tsv", sep="\t")
```

### Issue: "Low recall on validation set"
**Solution**: 
- Reduce blocking thresholds (Jaccard ≥ 0.5 → ≥ 0.4)
- Add more blocking stages (phonetic matching, abbreviation expansion)
- Check if candidate set includes all ground truth matches

### Issue: "Leaderboard score much lower than validation"
**Possible causes**:
- Validation split not representative
- Data distribution shift (France in test, not in train)
- Singletons poorly predicted
**Solution**: Re-evaluate on test with higher recall/lower threshold trade-off

### Issue: "Validation fails: Matched ID not in candidate set"
**Solution**: Bug in pipeline — ensure:
1. `candidate_pairs.tsv` is generated BEFORE model inference
2. Model only predicts IDs from candidate set
3. No post-processing removes IDs

---

## 📝 Logging & Debugging

All runs log to `logs/pipeline.log`. Check for:
- Data loading stats
- Blocking recall ceiling
- Feature generation time
- Model training progress
- F₀.₅ score improvements

Enable verbose logging:
```bash
python -m src.pipeline.main \
    --train-dir dataset/train \
    --test-dir dataset/test \
    --output-dir output/ \
    --verbose \
    --log-level DEBUG
```

---

## 📋 Team Member Responsibilities

| Member | Module | Deliverable | Hours |
|--------|--------|-------------|-------|
| **1** (Data Eng) | `src/data/` | Cleaned data, EDA report | 0-12 |
| **2** (Blocking) | `src/blocking/` | Candidate pairs, blocking metrics | 6-48 |
| **3** (ML) | `src/features/`, `src/model/` | Trained model, F₀.₅ score | 12-60 |
| **4** (Pipeline) | `src/pipeline/`, `src/validation/` | End-to-end runner, ZIP package | 24-72 |

**Daily Standups**: Hours 12, 36, 60

---

## 📤 Submission

### Leaderboard Upload (During Challenge)
```bash
# Upload this file to the Portal:
output/matching_results.tsv
```

### Final Submission Package (After Challenge)
```bash
# Create ZIP with this structure:
team_name_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md
```

**Create ZIP**:
```bash
zip -r team_name_submission.zip output/ code/ Documentation_template.md
```

---

## 📚 References & Resources

### String Similarity Libraries
- `fuzzywuzzy`: Fuzzy string matching (Levenshtein-based)
- `rapidfuzz`: Fast fuzzy matching (C++ backend)
- `textdistance`: Multiple distance metrics
- `thefuzz`: Modern fuzzy matching

### ML Libraries
- `xgboost`: Gradient boosting (fast, interpretable)
- `lightgbm`: Gradient boosting (speed optimized)
- `scikit-learn`: Classic ML models + utilities
- `optuna`: Hyperparameter optimization

### Utilities
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `tqdm`: Progress bars
- `loguru`: Better logging

### Learning Resources
- [Entity Resolution Overview](https://en.wikipedia.org/wiki/Record_linkage)
- [String Similarity Metrics](https://en.wikipedia.org/wiki/String_metric)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [F-Beta Score](https://en.wikipedia.org/wiki/F-score)

---

## 📄 License

This project uses models licensed under **Apache 2.0 or MIT**. See `LICENSE` file for details.

---

## 🤝 Contributing

For team members:
1. Create feature branch: `git checkout -b feature/blocking-v2`
2. Commit changes: `git commit -m "Improve blocking recall"`
3. Push to branch: `git push origin feature/blocking-v2`
4. Create Pull Request (peer review before merge)

**Code Style**: PEP 8, type hints encouraged, docstrings required.

---

## ❓ FAQ

**Q: Can we use pre-trained embeddings (BERT, word2vec)?**  
A: Yes, if the model is open-source (Apache/MIT licensed) and ≤8B parameters.

**Q: What if the test data includes a country not in training?**  
A: It does (France). Don't hard-code country filters; treat country as an open set.

**Q: How many times can we submit to the leaderboard?**  
A: Check challenge rules. Typically unlimited, but may have rate limits.

**Q: What's a good F₀.₅ score?**  
A: Depends on competition. Aim for >0.75 on validation split. Top teams usually achieve 0.80-0.90+.

**Q: Should we use different models for S2 and S3?**  
A: Try both unified (single model) and separate models. Usually unified is simpler & competitive.

---

## 📞 Support

For issues or questions:
- Check this README first (likely answered here)
- Review Jupyter notebooks (`notebooks/`) for examples
- Check logs in `logs/pipeline.log`
- Ask team members during daily standups

---

**Good luck with the challenge! 🚀**

*Last Updated: 2024 | Amazon ML Challenge*
