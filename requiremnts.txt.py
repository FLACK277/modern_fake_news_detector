# Write the requirements.txt file with modern dependencies
requirements_content = """
# Core ML and DL libraries
torch>=2.0.0
torchvision>=0.15.0
transformers>=4.35.0
datasets>=2.14.0
tokenizers>=0.14.0

# Modern transformer and NLP libraries
sentence-transformers>=2.2.2
accelerate>=0.24.0
peft>=0.6.0

# Hyperparameter optimization
optuna>=3.4.0
ray[tune]>=2.8.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pillow>=10.0.0

# API and deployment
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.4.0
pydantic-settings>=2.0.0

# MLOps and monitoring
mlflow>=2.8.0
wandb>=0.16.0
tensorboard>=2.15.0

# Configuration management
hydra-core>=1.3.0
omegaconf>=2.3.0
python-dotenv>=1.0.0

# Multimodal processing
timm>=0.9.0
opencv-python>=4.8.0
albumentations>=1.3.0

# Memory and performance optimization
psutil>=5.9.0
py-cpuinfo>=9.0.0

# Database and storage
sqlalchemy>=2.0.0
redis>=5.0.0

# Testing and development
pytest>=7.4.0
pytest-asyncio>=0.21.0
black>=23.0.0
isort>=5.12.0
"""

with open("modern_fake_news_detector/requirements.txt", "w") as f:
    f.write(requirements_content.strip())

print("Requirements.txt created successfully!")