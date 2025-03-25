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
- **V**: Voltage (Volts)  
- **I**: Current (Amperes)  
- **R**: Resistance (Ohms)

---

### ⚡ 2. Power (P) – Real Power
\[
P = V \cdot I \cdot \cos(\phi)
\]
- **P**: Active Power (Watts)  
- **ϕ**: Power factor angle (between voltage and current)

---

### 🔁 3. Reactive Power (Q)
\[
Q = V \cdot I \cdot \sin(\phi)
\]
- **Q**: Reactive Power (VARs)  
- Important for voltage regulation in the grid.

---

### 🔷 4. Apparent Power (S)
\[
S = \sqrt{P^2 + Q^2}
\]
- **S**: Apparent Power (VA)  
- Represents the combination of active and reactive power.

---

### 🌀 5. Power Angle (δ) in Synchronous Machines
\[
P = \frac{EV}{X} \cdot \sin(\delta)
\]
- **E**: Internal voltage of generator  
- **V**: Bus voltage  
- **X**: Reactance between buses  
- **δ**: Power angle  
- This equation governs the **synchronism and stability** of generators.

---

## 🏗️ Project Structure

