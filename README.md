# 🔍 Fake News Detection using Machine Learning & Deep Learning

A comprehensive fake news detection system that combines traditional machine learning and transformer models to classify news articles as fake or real with high accuracy.

## Vercel Deployment (Full-Stack App)

This repository includes a deployable full-stack app in `fake-news-detector/` and a root-level `vercel.json` so you can deploy directly from the repository root.

### What gets deployed

- Static frontend build from `fake-news-detector/frontend`
- Python API function from `fake-news-detector/backend/app.py`
- API routing via `/api/*` (for example `/api/predict` and `/api/health`)

### Step-by-step deployment

1. Install Vercel CLI (one-time):

```bash
npm i -g vercel
```

2. Login to Vercel:

```bash
vercel login
```

3. Confirm required model file exists:

```bash
ls fake-news-detector/backend/models/prediction_model.pkl
```

4. From repository root, run first deploy:

```bash
vercel
```

5. When prompted by CLI, use these answers:
- Set up and deploy: `Y`
- Scope: choose your account/team
- Link to existing project: `N` (or `Y` if reusing one)
- Project name: your preferred name
- In which directory is your code located: `./`
- Override settings: `N` (use `vercel.json` in repo)

6. Promote to production:

```bash
vercel --prod
```

### Deploy from Vercel Dashboard (alternative)

1. Import GitHub repository in Vercel.
2. Keep Root Directory as repository root.
3. Build and output settings should remain auto-detected from `vercel.json`.
4. Deploy.

### Verify deployment

After deploy, test:

- `GET /api/health`
- `POST /api/predict` with JSON body:

```json
{
   "text": "Some news content to classify"
}
```

The `.vercelignore` file excludes notebooks, Docker files, and cache folders to reduce upload size.

## 📊 Performance Metrics

| Model | Accuracy | F1-Score | Precision | Recall | ROC-AUC |
|-------|----------|----------|-----------|--------|---------|
| **Final Ensemble** | 99.22% | 99.22% | 98.90% | 99.56% | 99.96% |
| Traditional ML | 99.11% | 99.12% | 98.68% | 99.56% | 99.97% |
| Transformer | 77.22% | 75.33% | 82.15% | 69.56% | 85.64% |

Cross-validation results show robust performance with mean accuracy >97% across all models.

## ✨ Features

- **Hybrid Ensemble Architecture**: Combines traditional ML (LogisticRegression, RandomForest, GradientBoosting, XGBoost) with BERT-based transformers
- **Advanced Text Preprocessing**: URL/email removal, lemmatization, stop-word filtering, and punctuation handling
- **Cross-Validation**: 3-fold stratified cross-validation for robust performance estimation
- **Fast Training Mode**: Optimized settings for quick experimentation (5-10 minutes)
- **Real-time Predictions**: Predict any news article as fake or real with confidence scores
- **Visualization**: Confusion matrices and performance metrics

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection

# Install required packages
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### Dataset Setup

1. Download the datasets:
   - [Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) - Fake news articles
   - [True.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) - Real news articles

2. Place them in your `Downloads/` folder or update paths in `CFG` class:
   ```python
   fake_path : str = "path/to/Fake.csv"
   real_path : str = "path/to/True.csv"
   ```

### Usage

#### Training the Model

```python
# Initialize and train the detector
from fake_news_detector import FakeNewsDetector, DataManager

# Load and preprocess data
DM = DataManager()
DM.load()
DM.preprocess()

# Train the model
detector = FakeNewsDetector(DM.df)
detector.train()
```

#### Making Predictions

```python
# Predict a single article
article = """
Scientists have discovered a new species of dinosaur in Argentina.
The fossils indicate it lived 90 million years ago.
"""

result = detector.predict(article)
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.4f}")
print(f"Confidence: {result['confidence']:.4f}")
```

## 📦 Requirements

```
numpy>=1.17
pandas
scikit-learn
xgboost
tensorflow>=2.0
transformers>=4.46
torch>=2.1
accelerate>=0.26.0
matplotlib
seaborn
nltk
```

## 🏗️ Architecture

### Text Preprocessing Pipeline
1. Lowercase conversion
2. URL, email, mention, hashtag removal
3. HTML tag removal
4. Number removal
5. Punctuation removal (optional)
6. Tokenization
7. Lemmatization
8. Stop-word filtering

### Feature Extraction
- **TF-IDF Vectorization**: 5,000 features with bi-grams
- **Transformer Encoding**: BERT-tiny embeddings (256 max length)

### Model Ensemble
1. **Traditional ML Models**:
   - Logistic Regression (mean CV: 97.69%)
   - Random Forest (mean CV: 98.11%)
   - Gradient Boosting (mean CV: 99.28%)
   - XGBoost (mean CV: 99.44%)

2. **Transformer Model**:
   - prajjwal1/bert-tiny (4M parameters)
   - Fine-tuned for sequence classification

3. **Final Ensemble**:
   - Weighted average: 40% Traditional + 60% Transformer

## ⚙️ Configuration

Customize settings in the `CFG` dataclass:

```python
@dataclass
class CFG:
    sample_size: int = 5000          # Dataset size for fast testing
    max_vocab: int = 10000           # Max vocabulary size
    max_len: int = 256               # Max sequence length
    tfidf_max_features: int = 5000   # TF-IDF features
    model_name: str = "prajjwal1/bert-tiny"  # Transformer model
    epochs: int = 1                  # Training epochs
    batch_size: int = 16             # Batch size
    lr: float = 3e-5                 # Learning rate
    cv_folds: int = 3                # Cross-validation folds
```

## 📈 Results

### Confusion Matrix
The final ensemble achieves near-perfect classification with minimal false positives/negatives.

### Cross-Validation Performance
```
LogReg: 0.9769 (+/- 0.0089)
RF:     0.9811 (+/- 0.0114)
GB:     0.9928 (+/- 0.0008)
XGB:    0.9944 (+/- 0.0028)
```

### Final Validation (Unseen Data)
- **Accuracy**: 99.80%
- **Perfect Recall**: 100%
- **High Precision**: 99.60%

## 🎯 Use Cases

- **News Verification**: Automatically flag potentially fake news articles
- **Social Media Monitoring**: Detect misinformation in real-time
- **Educational Tools**: Help users understand fake news patterns
- **Research**: Benchmark for fake news detection research

## 🛠️ Troubleshooting

### Common Issues

**ImportError: accelerate not found**
```bash
pip install accelerate>=0.26.0
```

**NLTK data not found**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
```

**Memory issues**
- Reduce `sample_size` in CFG
- Reduce `tfidf_max_features`
- Use CPU instead of GPU for small datasets

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{fake_news_detection,
  author = {Your Name},
  title = {Fake News Detection using ML and Transformers},
  year = {2025},
  url = {https://github.com/yourusername/fake-news-detection}
}
```

## 🙏 Acknowledgments

- Dataset: [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Transformer Model: [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)
- Built with scikit-learn, transformers, and PyTorch

## 📧 Contact

For questions or feedback, please open an issue or contact [your-email@example.com]

---

**Note**: This is a fast-test version optimized for experimentation. For production deployment, consider using the full dataset and more training epochs.
