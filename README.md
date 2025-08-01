# BBOB Algorithm Selection Project

## Overview
This project helps select the best optimization algorithm for solving problems in the Black-Box Optimization Benchmarking (BBOB) dataset. It uses machine learning (ML) to predict which algorithm performs best for a given problem, based on its characteristics. The project includes data preprocessing, training two ML models (MLP and CNN), and evaluating their predictions to measure performance.

The goal is to make algorithm selection easier and faster for optimization tasks, especially for people working on complex problems where choosing the right algorithm is challenging.

---

## What is BBOB?
BBOB is a standard set of 24 mathematical problems used to test optimization algorithms. Each problem has different features (like dimensions and instances) and challenges (like being multimodal or noisy). This project analyzes 10 specific algorithms, such as `BIPOP-CMA-ES` and `DE-BFGS_voglis_noiseless`, to find the best one for each problem.

---

## Project Features
- **Data Preprocessing**: Cleans and prepares BBOB data for machine learning.
- **Machine Learning Models**:
  - **MLP (Multi-Layer Perceptron)**: A simple neural network that achieves 84.7% accuracy.
  - **CNN (Convolutional Neural Network)**: A more complex model that achieves 88.6% accuracy.
- **Performance Evaluation**: Measures how well the predicted algorithms perform using a metric called RELERT (Relative Expected Runtime).
- **Summaries**: Provides tables showing algorithm performance across problem types and dimensions.

---

## How It Works
1. **Preprocessing**: The project starts with raw BBOB data and processes it to create a clean dataset with 291 features and labels for the best algorithm per problem.
2. **Training**: Two models (MLP and CNN) are trained to predict the best algorithm based on problem features.
3. **Prediction**: The trained models predict algorithms for new problems.
4. **Evaluation**: The predictions are evaluated by checking the RELERT of the chosen algorithms, showing how close they are to the best possible performance.
5. **Summarization**: Results are summarized in tables to compare performance across problem types and dimensions.

---

## Project Structure
The project is organized into several Jupyter notebooks, each handling a specific task:

- **Preprocessing**:
  - `Code-01.ipynb` to `Code-07.ipynb`: Merge, clean, and prepare BBOB data, creating `normalized_dataset.csv` and `optimization_data_relert.csv`.
  - `T-01.ipynb`: Summarizes actual algorithm performance (RELERT) by problem type and dimension, saved to `RELERt_table_with_ALL.xlsx`.

- **Training**:
  - `Train.ipynb` (MLP version): Trains an MLP model, saves it as `mlp_model.pkl`, and evaluates it (results in `results.txt`).
  - `Train.ipynb` (CNN version): Trains a CNN model, saves it as `cnn_model.h5`, and evaluates it (results in `result.txt`).

- **Evaluation (MLP)**:
  - `1.ipynb`: Uses the MLP model to predict algorithms for a test dataset, saving results to `predictions.csv`.
  - `2.ipynb`: Combines predictions with RELERT values, saving to `predictions_with_relert.csv`.
  - `3.ipynb`: Summarizes RELERT of predicted algorithms by problem type and dimension, saving to `FFNew_RELERt_summary.xlsx`.

- **Evaluation (CNN)**:
  - `1.ipynb`: Uses the CNN model to predict algorithms, saving to `predictions.csv`.
  - `2.ipynb`: Merges predictions with RELERT values, saving to `predictions_with_relert.csv`.
  - `3.ipynb`: Summarizes RELERT of predicted algorithms, saving to `FFNew_RELERt_summary.xlsx`.

- **Key Files**:
  - `normalized_dataset.csv`: Processed dataset for training.
  - `optimization_data_relert.csv`: Dataset with RELERT values for all algorithms.
  - `results.txt`: MLP performance metrics (accuracy, confusion matrix, etc.).
  - `result.txt`: CNN performance metrics.
  - `predictions.csv`: Predicted algorithms for test data.
  - `predictions_with_relert.csv`: Predictions with corresponding RELERT values.
  - `RELERt_table_with_ALL.xlsx`: Summary of actual algorithm performance.
  - `FFNew_RELERt_summary.xlsx`: Summary of predicted algorithm performance.

---

## Getting Started

### Prerequisites
To run this project, you need:
- Python 3.8 or higher
- Jupyter Notebook
- Libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `imblearn`, `joblib`
- Google Drive access (for file paths in the notebooks)

Install the required libraries:
```bash
pip install pandas numpy scikit-learn tensorflow imblearn joblib
```

### Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/bbob-algorithm-selection.git
   cd bbob-algorithm-selection
   ```
2. **Upload Data**:
   - Place `optimization_data_relert.csv`, `normalized_dataset.csv`, `test_dataset.csv`, and `dataset.csv` in the appropriate directories (e.g., `Wail-Projet-F/MLP/Data` or `Wail-Projet-F/CNN/Data`).
   - Alternatively, update file paths in the notebooks to match your setup.

3. **Run the Notebooks**:
   - Start with preprocessing (`Code-01.ipynb` to `Code-07.ipynb`) to prepare data.
   - Train models using `Train.ipynb` (MLP or CNN version).
   - Run `T-01.ipynb` for a baseline summary of algorithm performance.
   - Run `1.ipynb`, `2.ipynb`, and `3.ipynb` (MLP or CNN version) to predict and evaluate results.

### Example Usage
To predict algorithms using the MLP model:
1. Run `1.ipynb` to generate `predictions.csv`.
2. Run `2.ipynb` to add RELERT values, creating `predictions_with_relert.csv`.
3. Run `3.ipynb` to summarize RELERT by problem type and dimension in `FFNew_RELERt_summary.xlsx`.

Compare the summary with `RELERt_table_with_ALL.xlsx` from `T-01.ipynb` to see how close the predictions are to the best algorithms.

---

## Results
- **MLP Model**:
  - Test Accuracy: 84.7%
  - Macro F1-Score: 0.84
  - Best at predicting: `s-CMA-ES_Gissler` (F1=0.99)
  - Weakest at predicting: `DE-BFGS_voglis_noiseless` (F1=0.62)
- **CNN Model**:
  - Test Accuracy: 88.6%
  - Macro F1-Score: 0.88
  - Best at predicting: `BIPOP-CMA-ES` (F1=0.97)
  - Weakest at predicting: `DE-BFGS_voglis_noiseless` (F1=0.69)
- The CNN model performs better overall, but both struggle with `DE-BFGS_voglis_noiseless` due to its high representation in the dataset.

---

## How to Interpret Results
- **Accuracy**: Shows how often the model picks the correct algorithm.
- **RELERT**: Measures how efficient the predicted algorithm is compared to the best possible algorithm (lower is better).
- **Summaries**:
  - `RELERt_table_with_ALL.xlsx`: Shows actual algorithm performance across problem types (e.g., functions 1–5) and dimensions.
  - `FFNew_RELERt_summary.xlsx`: Shows the performance of predicted algorithms, helping you see if the model’s choices are effective.

---

## Limitations
- The dataset is small (480 samples), which may limit model performance.
- Both models struggle with `DE-BFGS_voglis_noiseless`, possibly due to imbalanced data.
- The MLP model uses PCA, which may lose some information, while the CNN model’s padding may add noise.
- File paths are specific to Google Drive; you may need to adjust them.

---

## Future Improvements
- **More Data**: Add more BBOB problem instances to improve model accuracy.
- **Better Preprocessing**: Try different feature selection methods or oversampling techniques (e.g., ADASYN instead of SMOTE).
- **Model Tuning**: Adjust MLP/CNN parameters (e.g., layers, dropout) to reduce overfitting.
- **Evaluation**: Compare predicted RELERTs directly with optimal RELERTs to measure performance gaps.
- **Visualizations**: Add plots to `3.ipynb` to visualize RELERT trends.

---

## Contributing
We welcome contributions! To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature-branch`).
3. Make your changes and test them.
4. Submit a pull request with a clear description.

Please follow the code style in the notebooks and test all changes before submitting.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Contact
For questions or feedback, please open an issue on GitHub or contact [your-email@example.com].
