# ⚡ Stability by Design: Machine Learning Models for Smart Grid Predictions

Welcome to the repository for our comparative study on machine learning models used to predict **smart grid stability**. This project investigates how well various classification algorithms can distinguish between **stable** and **unstable** grid conditions based on system parameters.

---

## 🌐 Project Overview

The **Smart Grid** is a modernized electrical grid that uses digital communication and automation to manage electricity flow efficiently, integrating traditional power infrastructure with renewable energy sources, storage, and intelligent control systems.

However, with increased complexity comes increased vulnerability — **Grid Stability** is a critical concern to prevent cascading failures and blackouts.

This project applies **machine learning techniques** to:
- Predict whether the grid will remain stable or not
- Analyze performance of various classifiers
- Provide insights into the most effective features and models

---

## 🧠 Machine Learning Models Evaluated

We implemented and compared the performance of the following classifiers:

- 🌲 **Random Forest Classifier**
- 🌳 **Decision Tree Classifier**
- 💠 **Support Vector Machine (SVM)**
- 🚀 **XGBoost (Extreme Gradient Boosting)**

Each model was trained on the same dataset and evaluated using common classification metrics like Accuracy, Precision, Recall, and F1 Score.

---

## 📊 Dataset Description

The dataset used in this project contains numerical and categorical features derived from a simulated smart grid environment:

| Feature | Description |
|---------|-------------|
| `tau1`, `tau2`, `tau3`, `tau4` | Time constants for dynamic components in the grid |
| `p1`, `p2`, `p3`, `p4`         | Power demands at different grid nodes |
| `g1`, `g2`, `g3`, `g4`         | Generator efficiencies or output capacities |
| `stab`                         | Numerical stability index |
| `stabf`                        | Categorical label (`'stable'`, `'unstable'`) — used as the target class |

---

## 📈 Distributions

### 🔹 tau1, g1, and stab
The distributions of **tau1**, **g1**, and **stab** suggest a relatively uniform spread across their ranges:
- **tau1 and g1** appear evenly distributed, reflecting a variety of operating conditions.
- **stab** (stability index) is slightly skewed but still spans a broad range, indicating variation in grid behavior.

### 🔹 p1
The distribution of **p1** (power demand) shows more variability:
- It includes both low and high-demand situations, with a noticeable peak where most values fall.
- This highlights the model's exposure to diverse power demand levels, covering both supply and load scenarios.

---

## 🔍 Random Forest Classifier – Results

The **Random Forest** model achieved strong performance:

- **Accuracy:** 94.79%

### 📊 Class-wise Metrics

| Metric    | Stable | Unstable |
|-----------|--------|----------|
| Precision | 94%    | 95%      |
| Recall    | 91%    | 97%      |
| F1-Score  | 93%    | 96%      |

> 🔎 The model performs **slightly better in detecting 'unstable' conditions**, which is valuable for real-world applications where identifying risky grid states is crucial.

---

## 🌲 Feature Importance Analysis

Using the Random Forest model's built-in feature importance capability, we gain the following insights:

### 🕒 Time Constants (`tau1`, `tau2`, `tau3`, `tau4`)
- These are the **most influential features**.
- `tau1` is the most critical, indicating the strong role of component response time in grid stability.

### 🔋 Generator Capacities (`g1` to `g4`)
- Also significant, particularly `g3` and `g4`.
- Highlight the importance of generation strength in maintaining grid balance.

### ⚡ Power Demands (`p1` to `p4`)
- Have **less predictive power** but still contribute meaningfully.
- `p1` shows slightly more importance among them.

---

## 🚀 XGBoost Classifier – Results

The **XGBoost model** outperformed all others in the study:

- **Highest Accuracy** among all models
- Excellent **precision, recall, and F1-scores** for both classes
- Strong at capturing **non-linear interactions** between variables
- Efficiently handled feature-rich, complex data

> ✅ This makes **XGBoost** a preferred model for real-time and high-accuracy prediction tasks in critical systems like smart grids.

---

## ⚡ Understanding Grid Stability

### What is Grid Stability?

Grid stability refers to the **power system’s ability to maintain continuous and balanced operation** in the presence of disturbances such as:

- Sudden demand surges
- Generator trips
- Transmission line faults
- Variable renewable energy sources (solar/wind)

A stable grid ensures:
- Constant voltage and frequency levels
- No blackouts or service interruptions
- Reliable electricity delivery

---

## 🔣 Power System Equations and Fundamentals

The following fundamental electrical equations are used to understand the physical dynamics of the grid and can also serve as derived features in advanced modeling:

### 🔌 1. Ohm’s Law
\[
V = I \cdot R
\]

### ⚡ 2. Power (P) – Real Power
\[
P = V \cdot I \cdot \cos(\phi)
\]

### 🔁 3. Reactive Power (Q)
\[
Q = V \cdot I \cdot \sin(\phi)
\]

### 🔷 4. Apparent Power (S)
\[
S = \sqrt{P^2 + Q^2}
\]

### 🌀 5. Power Angle Equation (δ)
\[
P = \frac{EV}{X} \cdot \sin(\delta)
\]


---

## 🚀 Future Scope

As energy systems continue to evolve, the role of predictive analytics and intelligent decision-making will grow in importance. This project lays the foundation for several future directions in smart grid research and development:

### 🌐 1. Real-Time Stability Monitoring
Integrating the trained models into real-time grid management systems could enable **instantaneous decision-making** and **preventive control actions**, enhancing grid resilience and reducing the risk of blackouts.

### ⚡ 2. Renewable Energy Integration
As the penetration of intermittent renewable energy sources increases, stability challenges will become more dynamic. These models can be extended to **predict renewable-induced instabilities** and help grid operators **balance supply-demand intelligently**.

### 📉 3. Predictive Maintenance and Anomaly Detection
By coupling this system with sensor data from grid equipment, utilities could detect early warning signs of component failures or performance degradation, enabling **predictive maintenance strategies**.

### 🤖 4. Model Optimization with Deep Learning
Future iterations could include **deep learning architectures** (e.g., LSTM, CNN) for handling time-series data or sequential grid states, allowing for **temporal pattern recognition** and improved accuracy in dynamic environments.

### 🛰️ 5. Integration with IoT and Edge Computing
Deploying lightweight versions of these models on **IoT edge devices** could enable distributed and decentralized grid monitoring — reducing latency and improving system responsiveness.

### 🧪 6. Expansion to Multimodal Datasets
Incorporating other data sources like **weather conditions, historical fault logs, and energy pricing** could enrich the feature set, enabling **multi-factor stability prediction models**.


### About ME
Shoaib Ahmad
BS-EE 2018-22
Power System Analysis, Power Energy Optimzation


---

> This project is a step toward **intelligent energy systems** — making power grids not just smarter, but also safer, more sustainable, and proactive.


