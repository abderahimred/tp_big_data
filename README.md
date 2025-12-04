# 📊 Big Data Clustering: Comparative Analysis

> **Apache Spark MLlib vs Scikit-learn** - A comprehensive performance benchmark of K-Means clustering implementations

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-4.0.1-orange.svg)](https://spark.apache.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-yellow.svg)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Results Overview](#-results-overview)
- [Key Findings](#-key-findings)

---

## 🎯 Overview

This project presents a **comprehensive comparative analysis** of K-Means clustering implementations using **Apache Spark MLlib** and **Scikit-learn**. We evaluate both frameworks across multiple dimensions:

- ⏱️ **Execution Time**
- 💾 **Memory Consumption**
- 📈 **Scalability**
- 🎯 **Clustering Quality**
- 🛠️ **API Complexity**

### 🎓 Academic Context

- **Course:** Data Science
- **Institution:** Département TNI, Université Évry Paris-Saclay
- **Date:** December 2025
- **Project Type:** Practical Work (TP) - Clustering Analysis

---

## ✨ Features

### 🔬 Comprehensive Benchmarking
- Tests across **3 dataset sizes** (Small: <10k, Medium: 10k-100k, Large: >100k samples)
- Multiple **cluster configurations** (k = 3, 5, 10)
- **18 total experiments** (3 datasets × 3 k-values × 2 frameworks)

### 📊 Advanced Visualizations
- **Performance comparison charts** (execution time, memory usage)
- **Scalability analysis** (log-log plots)
- **Quality metrics** (Inertia, Silhouette Score)
- **PCA-based cluster visualization**
- **Hexbin plots** for large datasets (1M+ samples)

### 🎯 Quality Metrics
- **Inertia (Within-cluster Sum of Squares)**
- **Silhouette Score** (cluster separation)
- **Davies-Bouldin Index** (cluster compactness)

### 🛡️ Robust Implementation
- **Error handling** for dataset loading failures
- **Memory-safe** Silhouette calculation (sampling for large datasets)
- **Progress tracking** with informative output
- **Automatic result export** to CSV

---
## 🚀 Installation

### Prerequisites

- Python 3.12+ (tested on Python 3.13)
- Jupyter Notebook or JupyterLab
- At least 8GB RAM (16GB recommended for large dataset)

### Step 1: Clone the Repository

```bash
git clone https://github.com/abderahimred/tp_big_data.git
cd tp_big_data
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install pyspark scikit-learn matplotlib seaborn pandas numpy psutil ipywidgets jupyter
```

### Step 4: Enable Jupyter Widgets (for interactive features)

```bash
jupyter nbextension enable --py widgetsnbextension
```

---

## 📖 Usage

### Quick Start

1. **Launch Jupyter Notebook:**

```bash
jupyter notebook
```

2. **Open the notebook:**
   - Navigate to `tp_big_data.ipynb`
   - Click to open

3. **Run all cells:**
   - Menu: `Kernel` → `Restart & Run All`
   - Or use keyboard shortcut: `Shift + Enter` for each cell


### Execution Time

- **Small dataset (Wine):** ~5 seconds
- **Medium dataset (MNIST):** ~30 seconds
- **Large dataset (Synthetic):** ~5-15 minutes

**Total notebook runtime:** ~15-30 minutes (depending on hardware)

### Expected Output

The notebook will:
1. ✅ Load and preprocess 3 datasets
2. ✅ Run 18 clustering experiments
3. ✅ Generate performance comparison charts
4. ✅ Create cluster visualizations
5. ✅ Export results to CSV
6. ✅ Display comprehensive analysis

---

## 📊 Datasets

### 1. Wine Quality (Small Dataset)

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **Size:** ~1,600 samples
- **Features:** 11 physicochemical properties
- **Use Case:** Small-scale clustering

### 2. MNIST (Medium Dataset)

- **Source:** [Scikit-learn Built-in](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
- **Size:** 70,000 samples
- **Features:** 784 pixel values (28×28 images)
- **Use Case:** Medium-scale clustering

### 3. Synthetic Data (Large Dataset)

- **Source:** Generated using `sklearn.datasets.make_blobs`
- **Size:** 1,000,000 samples
- **Features:** 20 dimensions
- **True Clusters:** 10
- **Use Case:** Large-scale scalability testing

---

## 📈 Results Overview

### Performance Summary

| Dataset | Size | Scikit-learn | Spark MLlib | Winner |
|---------|------|--------------|-------------|--------|
| **Wine Quality** | 1.6k | ~0.02s | ~0.5s | 🏆 Scikit-learn (25× faster) |
| **MNIST** | 70k | ~0.8s | ~2.5s | 🏆 Scikit-learn (3× faster) |
| **Synthetic** | 1M | ~45s | ~120s | 🏆 Scikit-learn (2.7× faster) |

*Note: Results on single-machine local mode. Spark would excel on distributed clusters.*

### Quality Metrics

Both frameworks produce **comparable clustering quality**:
- Silhouette scores within ±0.02
- Inertia values within ±5%
- Minor differences due to initialization methods

---

## 🔑 Key Findings

### When to Use Scikit-learn ✅

- ✅ Data fits in memory (<10GB)
- ✅ Running on single machine
- ✅ Rapid prototyping needed
- ✅ Simple, intuitive API preferred
- ✅ Research and exploration

### When to Use Spark MLlib ✅

- ✅ Data exceeds single-machine memory
- ✅ Distributed cluster available
- ✅ Integration with big data ecosystem (HDFS, Kafka, etc.)
- ✅ Production pipelines with streaming data
- ✅ Horizontal scalability required

### Critical Insight 💡

> "For datasets under 10 million rows on a single machine, **Scikit-learn is the pragmatic choice**. Spark should be reserved for genuinely distributed, large-scale production systems."

---

## 🛠️ Technologies Used

### Core Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.13 | Programming language |
| **PySpark** | 4.0.1 | Distributed computing framework |
| **Scikit-learn** | 1.6.1 | Machine learning library |
| **Pandas** | 2.2.3 | Data manipulation |
| **NumPy** | 2.1.3 | Numerical computing |

### Visualization & Analysis

| Technology | Version | Purpose |
|------------|---------|---------|
| **Matplotlib** | 3.10.0 | Plotting library |
| **Seaborn** | 0.13.2 | Statistical visualization |
| **Jupyter** | - | Interactive notebooks |

### System Monitoring

| Technology | Version | Purpose |
|------------|---------|---------|
| **psutil** | 5.9.0 | System resource monitoring |
| **ipywidgets** | 8.1.5 | Interactive widgets |


---

## 🌟 Acknowledgments

- **Course Instructor:** Dr. Kenneth EZUKWOKE
- **Institution:** Département TNI, Université Évry Paris-Saclay
- **Datasets:** UCI Machine Learning Repository, Scikit-learn
- **Inspiration:** Apache Spark and Scikit-learn communities


<div align="center">


[⬆ Back to Top](#-big-data-clustering-comparative-analysis)

</div>
