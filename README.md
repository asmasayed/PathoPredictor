# PathoPredictor

A comprehensive pathogen prediction system using genomic LLM, host risk classification, parameter regression, and LSTM time series modeling.

## Project Structure

```
pathopredictor/
├── data/              # Data directories
├── models/            # Trained model checkpoints
├── src/               # Source code
├── notebooks/         # Jupyter notebooks for exploration
├── tests/             # Unit tests
├── requirements.txt   # Python dependencies
├── environment.yml    # Conda environment
└── run_pipeline.py    # Main pipeline script
```

## Installation

### Using conda (recommended)
```bash
conda env create -f environment.yml
conda activate pathopredictor
```

### Using pip
```bash
pip install -r requirements.txt
```

## Usage

Run the main pipeline:
```bash
python run_pipeline.py
```

## Modules

- **Module 1**: Genomic LLM (DN-BERT) for sequence analysis
- **Module 2**: Host risk classifier and parameter regressor
- **Module 3**: LSTM for time series prediction
- **Simulation**: SEIR model for epidemic simulation

## License

MIT License
