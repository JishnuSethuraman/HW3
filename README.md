# Time Series Anomaly Detection and Neural Networks

This repository contains various Python scripts focused on time series anomaly detection, synthetic data generation, and neural network models, including both handmade Convolutional Neural Networks (CNNs) and simple fully connected neural networks. Additionally, matrix profiles are utilized to detect anomalies in time series datasets.

## Project Structure

### Python Scripts

- **Discords.py**: This script is used to detect discords (anomalies) in time series data. Discords are observations in a time series that differ significantly from others.
  
- **MatrixProfile.py**: Implements the Matrix Profile technique to identify anomalies in time series data. Matrix Profile is an efficient algorithm to compute the similarity between subsequences in a time series.

- **CNN.py**: A handmade implementation of a Convolutional Neural Network (CNN), used for tasks such as image classification.

- **SyntheticDatasetGeneration.py**: This script is used to generate synthetic time series datasets. These datasets can be used for testing anomaly detection methods or training models.

- **externalDatasetMatrixProfile.py**: Runs matrix profile analysis on two external datasets (`Stranger_Things_Ratings.csv` and `gasoline.csv`). These datasets were downloaded from the internet and are used to showcase Matrix Profile on real-world data.

- **simpleNN.py**: Implements a simple neural network (fully connected) from scratch using NumPy. This script includes training logic and backpropagation for a basic neural network that can be used on synthetic data.

### Datasets

- **Stranger_Things_Ratings.csv**: Contains ratings data for the TV show *Stranger Things* over time. This dataset is used for time series analysis and anomaly detection using the Matrix Profile method.

- **gasoline.csv**: Contains gasoline-related data (e.g., prices, consumption) that is used for time series anomaly detection.

### Caching
- **__pycache__**: Contains Python bytecode for faster execution.

## Scripts Overview

### 1. Discords and Matrix Profile for Time Series Anomaly Detection
The two scripts `Discords.py` and `MatrixProfile.py` are used to detect anomalies (or discords) in time series datasets. They use Matrix Profile, a widely used technique for detecting anomalies by calculating the similarity between subsequences of time series.

### 2. Convolutional Neural Network (CNN)
`CNN.py` is a handmade implementation of a Convolutional Neural Network. This script provides a simplified version of CNN that can be modified and extended for various deep learning tasks.

### 3. Synthetic Dataset Generation
The `SyntheticDatasetGeneration.py` script allows you to generate synthetic time series data, which can be used for testing and evaluating the performance of anomaly detection algorithms.

### 4. Matrix Profile on External Datasets
`externalDatasetMatrixProfile.py` runs matrix profile analysis on two real-world datasets: `Stranger_Things_Ratings.csv` and `gasoline.csv`. This script demonstrates how matrix profiles can be applied to external datasets to identify patterns and anomalies.

### 5. Simple Neural Network (NN)
`simpleNN.py` is a lightweight implementation of a fully connected neural network using NumPy. The script initializes and trains a simple neural network on synthetic data with basic forward propagation, cost computation, backpropagation, and parameter updates.
