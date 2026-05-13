# PATHOPREDICTOR - DETAILED FILE EXPLANATIONS

This document provides comprehensive explanations of all files in the PathoPredictor project, their purposes, and how they work together.

---

## 📁 PROJECT STRUCTURE OVERVIEW

The project is organized into several main directories:
- **data/**: Raw, processed, and external data storage
- **models/**: Trained model checkpoints
- **src/**: Source code organized by functionality
- **notebooks/**: Jupyter notebooks for exploration and analysis
- **tests/**: Unit tests for validation

---

## 🔧 ROOT LEVEL FILES

### `run_pipeline.py`
**Purpose**: Main entry point for running the complete PathoPredictor pipeline.

**What it does**:
- Orchestrates the execution of all modules in sequence
- Imports configuration settings for each module
- Calls training functions for:
  - Module 1: Genomic LLM (DN-BERT)
  - Module 2: Host Risk Classifier
  - Module 2: Parameter Regressor
  - Module 3: LSTM Time Series Model
- Provides progress feedback during execution

**How to use**: Run `python run_pipeline.py` to execute the entire pipeline.

---

### `requirements.txt`
**Purpose**: Lists all Python package dependencies with minimum version requirements.

**Key dependencies**:
- `numpy`, `pandas`: Data manipulation
- `scikit-learn`: Machine learning utilities
- `torch`: PyTorch for deep learning models
- `scipy`: Scientific computing (for SEIR simulations)
- `fastapi`, `uvicorn`: API server framework
- `jupyter`: Notebook environment
- `matplotlib`, `seaborn`: Visualization

**How to use**: Install with `pip install -r requirements.txt`

---

### `environment.yml`
**Purpose**: Conda environment configuration file for reproducible setup.

**What it does**:
- Defines Python version (3.8)
- Specifies conda channels (conda-forge, defaults)
- Lists conda-installable packages
- Uses pip for packages not available in conda (like PyTorch)

**How to use**: Create environment with `conda env create -f environment.yml`

---

### `README.md`
**Purpose**: Project documentation and quick reference guide.

**Contents**:
- Project description
- Directory structure overview
- Installation instructions
- Usage examples
- Module descriptions

---

### `.gitignore`
**Purpose**: Tells Git which files/directories to exclude from version control.

**What it ignores**:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- IDE files (`.vscode/`, `.idea/`)
- Data files (keeps directory structure but ignores actual data)
- Model files (keeps directory structure but ignores trained models)
- Log files
- OS-specific files (`.DS_Store`, `Thumbs.db`)

---

### `SETUP_INSTRUCTIONS.txt`
**Purpose**: Step-by-step guide for setting up and running the project.

**Contents**: Detailed instructions for installation, running pipelines, starting API, running tests, etc.

---

## 📂 DATA DIRECTORY STRUCTURE

### `data/raw/`
**Purpose**: Storage for raw, unprocessed input data.

**Subdirectories**:
- **`genomic_fasta/`**: FASTA format genomic sequence files
- **`host_metadata/`**: Host organism metadata (species, age, location, etc.)
- **`phenotype_data/`**: Phenotypic characteristics data
- **`time_series/`**: Time-series data for epidemic tracking

**Note**: Contains `.gitkeep` files to preserve empty directory structure in Git.

---

### `data/processed/`
**Purpose**: Storage for processed/cleaned data ready for model training.

**Subdirectories**:
- **`module1/`**: Processed data for genomic LLM
- **`module2/`**: Processed data for classifier/regressor
- **`module3/`**: Processed data for LSTM
- **`module4/`**: Additional processed data storage

---

### `data/external/`
**Purpose**: External datasets or reference data from third-party sources.

---

## 🤖 MODELS DIRECTORY

### `models/module1_dnbert/`
**Purpose**: Stores trained DN-BERT model checkpoints and weights.

### `models/module2_classifier/`
**Purpose**: Stores trained host risk classifier model files.

### `models/module2_regressor/`
**Purpose**: Stores trained parameter regressor model files.

### `models/module3_lstm/`
**Purpose**: Stores trained LSTM model checkpoints.

---

## 💻 SOURCE CODE (`src/`)

### CONFIGURATION MODULE (`src/config/`)

#### `config.py`
**Purpose**: Centralized configuration management for the entire project.

**Key Components**:
- **Data paths**: Defines paths to raw, processed, and external data directories
- **Model paths**: Location for saving trained models
- **Module configurations**: Hyperparameters for each module:
  - `MODULE1_CONFIG`: DN-BERT settings (batch_size=32, learning_rate=1e-4)
  - `MODULE2_CLASSIFIER_CONFIG`: Classifier settings (batch_size=64, learning_rate=1e-3)
  - `MODULE2_REGRESSOR_CONFIG`: Regressor settings (batch_size=64, learning_rate=1e-3)
  - `MODULE3_CONFIG`: LSTM settings (batch_size=32, learning_rate=1e-3, sequence_length=100)
- **SEIR parameters**: Default epidemic model parameters (beta, gamma, sigma)

**Why it's important**: Single source of truth for all configuration, making it easy to adjust hyperparameters without modifying code.

---

### PREPROCESSING MODULE (`src/preprocessing/`)

#### `fasta_parser.py`
**Purpose**: Parses FASTA format files containing genomic sequences.

**What it does**:
- Reads FASTA files line by line
- Identifies sequence headers (lines starting with `>`)
- Extracts sequence IDs and corresponding DNA/RNA sequences
- Returns a dictionary mapping sequence IDs to sequences

**Example usage**:
```python
sequences = parse_fasta("data/raw/genomic_fasta/sequences.fasta")
# Returns: {"seq1": "ATCGATCG...", "seq2": "GCATGCAT..."}
```

**Why needed**: FASTA is the standard format for genomic data; this parser converts it to a Python-friendly format.

---

#### `sequence_tokenizer.py`
**Purpose**: Converts genomic sequences into numerical tokens for neural network input.

**What it does**:
- Implements a tokenizer class with vocabulary management
- `fit()`: Learns vocabulary from training sequences
- `encode()`: Converts sequences to token IDs
- `decode()`: Converts token IDs back to sequences

**Why needed**: Neural networks require numerical input, not raw DNA sequences. This bridges that gap.

**Note**: Currently a placeholder - needs implementation based on chosen tokenization strategy (k-mer, character-level, etc.).

---

#### `host_label_encoder.py`
**Purpose**: Encodes categorical host metadata into numerical labels.

**What it does**:
- Uses scikit-learn's LabelEncoder
- Converts categorical labels (e.g., host species names) to integers
- Supports inverse transformation (integers back to labels)

**Example**:
- Input: ["human", "mouse", "human"] → Output: [0, 1, 0]
- Inverse: [0, 1, 0] → ["human", "mouse", "human"]

**Why needed**: Machine learning models require numerical features, not text labels.

---

#### `phenotype_builder.py`
**Purpose**: Constructs phenotype feature vectors from raw phenotype data.

**What it does**:
- Reads phenotype data from CSV files
- Performs feature engineering (creating derived features)
- Returns processed DataFrame ready for model training

**Why needed**: Raw phenotype data often needs transformation into meaningful features for prediction.

---

#### `time_series_cleaner.py`
**Purpose**: Cleans and preprocesses time-series data.

**What it does**:
- Handles missing values (forward fill)
- Removes outliers
- Normalizes data if needed
- Returns cleaned DataFrame

**Why needed**: Real-world time-series data is often noisy and incomplete; cleaning improves model performance.

---

### MODULE 1: GENOMIC LLM (`src/module1_genomic_llm/`)

#### `train_dnbert.py`
**Purpose**: Training script for DN-BERT (DNA-BERT) model.

**What it does**:
- Loads genomic sequence data
- Initializes DN-BERT model architecture
- Trains the model on sequences
- Saves trained model checkpoints

**DN-BERT**: A transformer-based language model adapted for DNA sequences, similar to BERT but for genomics.

**Why needed**: Learns representations of genomic sequences that capture biological patterns.

---

#### `generate_mutations.py`
**Purpose**: Generates mutated sequences using the trained genomic LLM.

**What it does**:
- Takes a reference sequence
- Uses the trained model to predict likely mutations
- Generates multiple mutated variants
- Returns list of mutated sequences

**Why needed**: Understanding mutation patterns helps predict pathogen evolution and virulence.

---

#### `utils.py`
**Purpose**: Utility functions for Module 1.

**Functions**:
- `load_model()`: Loads saved model from disk
- `save_model()`: Saves model to disk

**Why needed**: Reusable helper functions for model management.

---

### MODULE 2: CLASSIFIER (`src/module2_classifier/`)

#### `model.py`
**Purpose**: Defines the neural network architecture for host risk classification.

**Architecture**:
- **Input layer**: Takes feature vector of size `input_dim`
- **Hidden layer 1**: Fully connected layer (input_dim → hidden_dim, default 128)
- **Dropout**: 20% dropout for regularization
- **Hidden layer 2**: Fully connected layer (hidden_dim → hidden_dim/2)
- **Dropout**: Another 20% dropout
- **Output layer**: Fully connected layer (hidden_dim/2 → num_classes, default 3)

**Activation**: ReLU (Rectified Linear Unit) for hidden layers

**Why this architecture**: 
- Dropout prevents overfitting
- Two hidden layers allow learning complex patterns
- Output size matches number of risk categories (low, medium, high)

---

#### `train_classifier.py`
**Purpose**: Training script for the host risk classifier.

**What it does**:
- Loads preprocessed features (from genomic LLM + host metadata)
- Creates model instance
- Defines loss function (likely CrossEntropyLoss for classification)
- Trains model with optimizer (likely Adam)
- Validates on test set
- Saves trained model

**Why needed**: Trains the model to predict host risk levels based on genomic and host features.

---

#### `predict_host_risk.py`
**Purpose**: Makes predictions using the trained classifier.

**What it does**:
- Loads trained model
- Takes input features
- Runs forward pass through model
- Returns predicted risk level (low/medium/high)

**Why needed**: Inference function for making predictions on new data.

---

### MODULE 2: REGRESSOR (`src/module2_regressor/`)

#### `model.py`
**Purpose**: Defines neural network architecture for SEIR parameter regression.

**Architecture**: Similar to classifier but outputs continuous values instead of classes.

**Output**: 3 values representing:
- `beta`: Transmission rate
- `gamma`: Recovery rate  
- `sigma`: Incubation rate

**Why needed**: Predicts epidemic model parameters from genomic/host features.

---

#### `train_regressor.py`
**Purpose**: Training script for parameter regressor.

**What it does**:
- Similar to classifier training but uses regression loss (MSE/MAE)
- Trains model to predict continuous parameter values
- Validates predictions against known parameters

**Why needed**: Learns to estimate epidemic parameters from pathogen characteristics.

---

#### `predict_parameters.py`
**Purpose**: Predicts SEIR model parameters.

**What it does**:
- Takes input features
- Returns dictionary with predicted beta, gamma, sigma values
- These parameters are used in SEIR simulations

**Why needed**: Provides parameters for epidemic modeling.

---

### MODULE 3: LSTM (`src/module3_lstm/`)

#### `model.py`
**Purpose**: Defines LSTM (Long Short-Term Memory) architecture for time-series prediction.

**Architecture**:
- **LSTM layers**: 2 layers with hidden dimension 64
- **Batch first**: Input format is (batch, sequence, features)
- **Fully connected layer**: Maps LSTM output to single value
- **Output**: Beta adjustment value (how transmission rate changes over time)

**Why LSTM**: 
- Captures temporal dependencies in time-series data
- Can learn long-term patterns
- Ideal for sequential epidemic data

**Why needed**: Predicts how transmission rate (beta) changes over time based on historical data.

---

#### `train_lstm.py`
**Purpose**: Training script for LSTM model.

**What it does**:
- Loads time-series data
- Creates sequences of fixed length (from config)
- Trains LSTM to predict beta adjustments
- Uses sequence-to-one prediction (many time steps → single output)

**Why needed**: Trains model to learn temporal patterns in epidemic data.

---

#### `predict_beta_adjustment.py`
**Purpose**: Predicts beta parameter adjustments from time-series.

**What it does**:
- Takes historical time-series data
- Predicts how beta should be adjusted
- Returns adjustment value

**Why needed**: Dynamically adjusts SEIR model parameters based on recent trends.

---

### SIMULATION MODULE (`src/simulation/`)

#### `seir_model.py`
**Purpose**: Implements the SEIR epidemic model.

**SEIR Model Explanation**:
- **S (Susceptible)**: People who can be infected
- **E (Exposed)**: Infected but not yet infectious
- **I (Infected)**: Currently infectious
- **R (Recovered)**: Recovered and immune

**Differential Equations**:
- `dS/dt = -βSI/N`: Susceptibles decrease when they contact infected
- `dE/dt = βSI/N - σE`: Exposed increase from contacts, decrease as they become infectious
- `dI/dt = σE - γI`: Infected increase from exposed, decrease as they recover
- `dR/dt = γI`: Recovered increase as infected recover

**Parameters**:
- `β (beta)`: Transmission rate
- `γ (gamma)`: Recovery rate
- `σ (sigma)`: Incubation rate (E → I)
- `N`: Total population

**Functions**:
- `seir_model()`: Defines the differential equations
- `simulate_seir()`: Solves equations using scipy's odeint

**Why needed**: Simulates epidemic spread to predict future cases and plan interventions.

---

#### `dynamic_seir.py`
**Purpose**: Extended SEIR model with time-varying transmission rate.

**Key Difference**: Beta (transmission rate) can change over time via a function.

**What it does**:
- Takes `beta_func` instead of constant beta
- Allows modeling interventions (lockdowns, vaccines) that change transmission
- More realistic than constant-parameter model

**Why needed**: Real epidemics have changing transmission rates due to interventions and behavior changes.

---

#### `parameters.py`
**Purpose**: Manages SEIR model parameters.

**What it does**:
- `SEIRParameters` class stores beta, gamma, sigma
- `update_beta()`: Updates transmission rate dynamically
- `to_dict()`: Converts to dictionary format

**Why needed**: Centralized parameter management for simulations.

---

### API MODULE (`src/api/`)

#### `main.py`
**Purpose**: FastAPI web service for making predictions via HTTP.

**Endpoints**:
- **GET `/`**: Root endpoint, returns API information
- **POST `/predict`**: Main prediction endpoint

**Request Format** (for `/predict`):
```json
{
  "sequence": "ATCGATCG...",
  "host_metadata": {"species": "human", "age": 30},
  "time_series": [1, 5, 10, 20, ...]
}
```

**Response**: Prediction results

**Features**:
- Automatic API documentation at `/docs` (Swagger UI)
- Alternative docs at `/redoc`
- Request validation using Pydantic models
- Can be deployed as a microservice

**Why needed**: Allows other applications to use PathoPredictor predictions via REST API.

---

### UTILITIES MODULE (`src/utils/`)

#### `logger.py`
**Purpose**: Sets up logging system for the project.

**What it does**:
- Creates logger with configurable name and level
- Adds file handler (logs to file)
- Adds console handler (logs to terminal)
- Formats log messages with timestamp, name, level, and message

**Usage**:
```python
logger = setup_logger("module1", "logs/module1.log")
logger.info("Training started")
```

**Why needed**: Centralized logging helps debug issues and track execution.

---

#### `helpers.py`
**Purpose**: General helper utility functions.

**Functions**:
- `ensure_dir()`: Creates directory if it doesn't exist
- `load_json()`: Loads JSON file
- `save_json()`: Saves data to JSON file

**Why needed**: Common operations used across multiple modules.

---

#### `metrics.py`
**Purpose**: Calculates evaluation metrics for models.

**Functions**:
- **`calculate_classification_metrics()`**: 
  - Accuracy: Percentage of correct predictions
  - Precision: True positives / (True positives + False positives)
  - Recall: True positives / (True positives + False negatives)
  - F1 Score: Harmonic mean of precision and recall

- **`calculate_regression_metrics()`**:
  - MSE (Mean Squared Error): Average squared difference
  - MAE (Mean Absolute Error): Average absolute difference
  - RMSE (Root Mean Squared Error): Square root of MSE
  - R² Score: Coefficient of determination (how well model fits)

**Why needed**: Evaluates model performance objectively.

---

## 📓 NOTEBOOKS (`notebooks/`)

### `exploration_module1.ipynb`
**Purpose**: Jupyter notebook for exploring Module 1 (Genomic LLM).

**Typical uses**:
- Visualizing genomic sequences
- Testing tokenization
- Analyzing model outputs
- Experimenting with hyperparameters

---

### `exploration_module2.ipynb`
**Purpose**: Jupyter notebook for exploring Module 2 (Classifier/Regressor).

**Typical uses**:
- Feature analysis
- Model performance visualization
- Parameter sensitivity analysis
- Error analysis

---

### `exploration_module3.ipynb`
**Purpose**: Jupyter notebook for exploring Module 3 (LSTM).

**Typical uses**:
- Time-series visualization
- Sequence analysis
- Temporal pattern identification
- Beta adjustment analysis

---

### `experiments.ipynb`
**Purpose**: General experimentation notebook.

**Typical uses**:
- Testing new ideas
- Quick prototyping
- Data exploration
- Ad-hoc analysis

---

## 🧪 TESTS (`tests/`)

### `test_preprocessing.py`
**Purpose**: Unit tests for preprocessing functions.

**Tests**:
- FASTA parsing correctness
- Label encoder encoding/decoding
- Data cleaning functions

**Why needed**: Ensures preprocessing works correctly before training models.

---

### `test_seir.py`
**Purpose**: Unit tests for SEIR model simulation.

**Tests**:
- Simulation produces correct output shape
- Population conservation (S + E + I + R = N at all times)
- Parameter validation

**Why needed**: Ensures epidemic simulations are mathematically correct.

---

### `test_models.py`
**Purpose**: Unit tests for model architectures.

**Tests**:
- Model forward pass produces correct output shapes
- All three model types (classifier, regressor, LSTM)
- Input/output dimension validation

**Why needed**: Ensures models are constructed correctly before training.

---

## 🔄 HOW IT ALL WORKS TOGETHER

### Complete Pipeline Flow:

1. **Data Input** → Raw data placed in `data/raw/` directories

2. **Preprocessing** → 
   - `fasta_parser.py` reads genomic sequences
   - `sequence_tokenizer.py` converts to tokens
   - `host_label_encoder.py` encodes metadata
   - `phenotype_builder.py` creates features
   - `time_series_cleaner.py` cleans time-series data
   - Processed data saved to `data/processed/`

3. **Module 1 Training** →
   - `train_dnbert.py` trains genomic LLM
   - Model learns sequence representations
   - Saved to `models/module1_dnbert/`

4. **Module 2 Training** →
   - `train_classifier.py` trains risk classifier
   - `train_regressor.py` trains parameter regressor
   - Models saved to respective `models/` directories

5. **Module 3 Training** →
   - `train_lstm.py` trains time-series model
   - Model learns temporal patterns
   - Saved to `models/module3_lstm/`

6. **Prediction** →
   - Use trained models to make predictions
   - `predict_host_risk.py` → Risk level
   - `predict_parameters.py` → SEIR parameters
   - `predict_beta_adjustment.py` → Beta adjustments

7. **Simulation** →
   - `seir_model.py` or `dynamic_seir.py` simulates epidemic
   - Uses predicted parameters
   - Outputs future case predictions

8. **API** →
   - `main.py` serves predictions via HTTP
   - Other applications can query predictions

9. **Evaluation** →
   - `metrics.py` calculates performance metrics
   - Tests validate correctness

---

## 📝 SUMMARY

This project implements a complete pathogen prediction pipeline that:
- Analyzes genomic sequences using deep learning
- Predicts host risk levels
- Estimates epidemic parameters
- Models disease spread using SEIR equations
- Provides predictions via web API

Each file has a specific role in this pipeline, working together to create a comprehensive pathogen analysis and prediction system.
