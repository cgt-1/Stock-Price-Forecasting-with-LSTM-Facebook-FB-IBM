# Stock Price Forecasting with LSTM: Facebook (FB) & IBM

A deep learning project that forecasts **next-day stock closing prices** for Facebook (FB) and IBM using **Long Short-Term Memory (LSTM)** networks built with TensorFlow/Keras.

The project compares a **baseline LSTM** with a modified, higher-capacity architecture across two datasets with different amounts of historical data: approximately **8 years of FB data** and **58 years of IBM data**.

---

## 📌 Project Overview

Stock prices are time series where observations depend on previous market behavior, making recurrent neural networks such as LSTMs a natural model to explore.

The models use the previous **5 business days of closing prices** to predict the next day's closing price. The final year of each dataset is held out as a chronological test set.

The project covers:

* Time-series data inspection and preprocessing
* Handling missing business days
* Chronological train/test splitting
* Feature scaling without data leakage
* Sliding-window creation
* LSTM model development
* Model comparison and evaluation
* Analysis of model convergence and overfitting

---

## 📂 Dataset

Two historical stock datasets are used:

| Stock   |   Rows | Date Range              | Training Data |
| ------- | -----: | ----------------------- | ------------: |
| **FB**  |  1,980 | 2012-05-18 → 2020-04-01 |      ~8 years |
| **IBM** | 14,663 | 1962-01-02 → 2020-04-01 |     ~58 years |

Each dataset contains:

`Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

Only **`Date` and `Close`** are used for modeling.

---

## 🧹 Data Preparation

Several preprocessing steps were applied to ensure the time-series data was suitable for LSTM training.

| Step                    | Approach                                                                                 |
| ----------------------- | ---------------------------------------------------------------------------------------- |
| Missing business days   | Reindexed to a complete business-day calendar and filled gaps using linear interpolation |
| Train/test split        | Final year of data reserved as the test set                                              |
| Scaling                 | `MinMaxScaler` fitted **only on training data**                                          |
| Sliding window          | Previous 5 days → next-day prediction                                                    |
| Validation              | Final 10% of training windows used as chronological validation data                      |
| Data leakage prevention | No shuffling and no fitting transformations on test data                                 |

The resulting datasets contained:

| Stock | Train Rows | Test Rows | Train Windows | Validation Windows |
| ----- | ---------: | --------: | ------------: | -----------------: |
| FB    |      1,791 |       263 |         1,607 |                179 |
| IBM   |     14,934 |       263 |        13,436 |              1,493 |

IBM therefore provides roughly **8× more training windows** than FB.

---

## 🤖 LSTM Models

Both models use an input shape of **5 time steps × 1 feature**.

|                | Baseline            | Modified                       |
| -------------- | ------------------- | ------------------------------ |
| Architecture   | LSTM(50) → Dense(1) | LSTM(64) → Dense(8) → Dense(1) |
| Optimizer      | Adam, LR = 0.001    | Adam, LR = 0.0005              |
| Loss           | MSE                 | MSE                            |
| Maximum Epochs | 100                 | 120                            |
| Batch Size     | 32                  | 32                             |
| Regularization | EarlyStopping       | EarlyStopping                  |
| Parameters     | ~10,450             | ~17,400                        |

The modified model increases the number of LSTM units and adds an additional dense layer, providing greater model capacity.

EarlyStopping was used to restore the best validation weights and reduce unnecessary training.

---

## 📊 Results

Performance was evaluated on the **held-out final year** after converting predictions back to the original USD scale.

| Stock | Model        |     RMSE |      MAE |      MAPE |
| ----- | ------------ | -------: | -------: | --------: |
| FB    | **Baseline** | **4.27** | **3.20** | **1.71%** |
| FB    | Modified     |   146.93 |   146.27 |    76.45% |
| IBM   | **Baseline** | **3.26** | **2.53** | **1.90%** |
| IBM   | Modified     |     3.73 |     2.64 |     2.00% |

### Key Findings

* **The baseline LSTM performed better on both stocks.**
* The FB baseline achieved a **1.71% MAPE**, with an average absolute error of approximately **$3.20** per prediction.
* The modified FB model failed to learn effectively, producing nearly flat predictions around the average price level.
* The modified IBM model showed signs of **early overfitting**, with validation loss increasing almost immediately.
* The IBM baseline performed best around epoch 29 based on validation loss.
* IBM's much longer historical dataset provides more training data, but older market behavior may not represent the conditions of the final test period.

### Main Takeaway

Increasing model complexity did **not** automatically improve forecasting performance. In this experiment, the simpler baseline architecture generalized better on both datasets.

---

## 🧠 What This Project Demonstrates

This project highlights several important considerations when working with deep learning for time-series forecasting:

* Maintaining chronological order during data splitting
* Avoiding data leakage when scaling time-series data
* Creating overlapping sliding windows for sequence prediction
* Using validation data chronologically
* Monitoring model convergence and overfitting
* Evaluating predictions in the original scale
* Comparing model complexity against generalization performance

---

## 🛠️ Tech Stack

* **Python**
* **TensorFlow / Keras** — LSTM models and training
* **scikit-learn** — scaling and evaluation metrics
* **pandas & NumPy** — data processing
* **Matplotlib & Seaborn** — visualization

---

## 📁 Project Structure

```text
├── StockPredictionLSTM.ipynb
├── FB.csv
├── IBM.csv
└── README.md
```

---

## 🔮 Future Improvements

* Perform a broader hyperparameter search for units, learning rate, window size, and dropout.
* Compare LSTM with **GRU** and stacked LSTM architectures.
* Incorporate additional market features such as `Open`, `High`, `Low`, and `Volume`.
* Train IBM using a more recent historical period to reduce the impact of outdated market regimes.
* Compare against a **naive forecasting baseline**, such as predicting tomorrow's price as today's price.
* Evaluate model stability across multiple random seeds.

---

## 📚 References

* [IBM Stock Splits and Stock Dividends](https://www.ibm.com/investor/att/pdf/IBM-Stock-Splits-and-Stock-Dividends.pdf)
* [TensorFlow / Keras Documentation](https://www.tensorflow.org/)
* [scikit-learn Documentation](https://scikit-learn.org/)
