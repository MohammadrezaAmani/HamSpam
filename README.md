# Persian Email Spam Detection using Advanced Machine Learning Algorithms

Welcome to the **Persian Email Spam Detection** repository! 📧🚫

This project leverages the power of cutting-edge Machine Learning algorithms to effectively identify and classify email spam written in Persian. We employ two sophisticated word embedding algorithms for feature representation:

1. **TF-IDF (Term Frequency-Inverse Document Frequency):**
   - A powerful algorithm capturing the importance of words in the document.

2. **Frequency-based Word Embedding:**
   - Utilizing word frequency to understand the underlying patterns in the text.

For text preprocessing, we rely on the robust Hazm library, ensuring that our data is well-prepared for the subsequent machine learning pipeline.

## Implemented Classification Algorithms

Our project implements six state-of-the-art classification algorithms:

1. **K-Nearest Neighbors (KNN):**
   - A non-parametric method for classification based on the similarity of data points.

2. **Logistic Regression:**
   - A linear model for binary classification tasks, providing a solid baseline for comparison.

3. **Decision Tree:**
   - A tree-like model that makes decisions based on the input features, offering interpretability.

4. **Random Forest:**
   - An ensemble of decision trees, enhancing the robustness and accuracy of the model.

5. **Naive Bayes:**
   - A probabilistic model that assumes independence among features, suitable for text classification.

6. **Support Vector Machine (SVM):**
   - Although not uploaded yet, SVM will be incorporated for its effectiveness in high-dimensional spaces.

## Preprocessing with Hazm

The preprocessing stage is a crucial step in our pipeline, and we employ the Hazm library to handle tasks such as stemming and tokenization. This ensures that our input data is cleaned and ready for feature extraction.

## Performance Visualization

### Accuracy Plot

Check out our accuracy plot to get a visual representation of the performance across different algorithms.

![Accuracy Plot](https://github.com/parvvaresh/email-spam-detection/blob/main/src/plots/accuracy.png)

### Accuracy Table

Here's a detailed table showcasing the accuracy of each algorithm with both TF-IDF and frequency-based word embedding:

| Algorithm              | Accuracy with TF-IDF | Accuracy with Frequency Word |
| ---------------------- | -------------------- | ----------------------------- |
| Logistic Regression    | 43.5%                | 97.5%                         |
| Decision Tree          | 90.0%                | 93.0%                         |
| Random Forest          | 92.5%                | 95.5%                         |
| Naive Bayes            | 43.5%                | 75.0%                         |
| K-Nearest Neighbors    | 92.5%                | 96.0%                         |

Feel free to explore the code and experiment with different algorithms and embeddings to enhance the performance of our Persian Email Spam Detection system. Happy coding! 🚀✨
