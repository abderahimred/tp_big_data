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
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Results Overview](#-results-overview)
- [Key Findings](#-key-findings)
- [Technologies Used](#-technologies-used)
- [How to Upload to GitHub](#-how-to-upload-to-github)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project presents a **comprehensive comparative analysis** of K-Means clustering implementations using **Apache Spark MLlib** and **Scikit-learn**. We evaluate both frameworks across multiple dimensions:

- ⏱️ **Execution Time**
- 💾 **Memory Consumption**
- 📈 **Scalability**
- 🎯 **Clustering Quality**
- 🛠️ **API Complexity**

### 🎓 Academic Context

- **Course:** Big Data
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

## 📁 Project Structure

```
tp_big_data/
│
├── README.md                          # This file
├── tp_big_data.ipynb                  # Main Jupyter Notebook
├── notebook_content.md                # Full notebook content (for reference)
├── spark_optimization_fix.md          # Spark task size optimization guide
│
├── results/                           # Generated results (created after running)
│   ├── kmeans_benchmark_results_*.csv
│   └── figures/
│       ├── execution_time_comparison.png
│       ├── scalability_analysis.png
│       ├── memory_usage.png
│       └── cluster_visualizations.png
│
└── requirements.txt                   # Python dependencies
```

---

## 🚀 Installation

### Prerequisites

- Python 3.12+ (tested on Python 3.13)
- Jupyter Notebook or JupyterLab
- At least 8GB RAM (16GB recommended for large dataset)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/tp_big_data.git
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

## 📤 How to Upload to GitHub

### Method 1: Using GitHub Desktop (Easiest)

1. **Install GitHub Desktop:**
   - Download from [desktop.github.com](https://desktop.github.com/)

2. **Create a new repository:**
   - Open GitHub Desktop
   - Click `File` → `New Repository`
   - Name: `tp_big_data`
   - Local Path: Select `/Users/rahim/Desktop/tp_big_data`
   - Click `Create Repository`

3. **Commit your files:**
   - Check all files in the left panel
   - Add commit message: "Initial commit: Big Data Clustering Project"
   - Click `Commit to main`

4. **Publish to GitHub:**
   - Click `Publish repository`
   - Choose public or private
   - Click `Publish Repository`

### Method 2: Using Command Line (Terminal)

1. **Navigate to your project:**

```bash
cd /Users/rahim/Desktop/tp_big_data
```

2. **Initialize Git repository:**

```bash
git init
```

3. **Create `.gitignore` file:**

```bash
cat > .gitignore << EOF
# Jupyter Notebook checkpoints
.ipynb_checkpoints/
__pycache__/
*.pyc

# Results and cache
results/
*.csv

# Virtual environment
venv/
env/

# System files
.DS_Store
EOF
```

4. **Add all files:**

```bash
git add .
```

5. **Commit:**

```bash
git commit -m "Initial commit: Big Data Clustering Comparative Analysis"
```

6. **Create repository on GitHub:**
   - Go to [github.com](https://github.com)
   - Click the `+` icon → `New repository`
   - Repository name: `tp_big_data`
   - Description: "Comparative analysis of K-Means clustering: Apache Spark vs Scikit-learn"
   - Choose Public or Private
   - **DO NOT** initialize with README (you already have one)
   - Click `Create repository`

7. **Link and push:**

```bash
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/tp_big_data.git

git branch -M main
git push -u origin main
```

### Method 3: Upload via GitHub Web Interface

1. **Create new repository on GitHub:**
   - Go to [github.com/new](https://github.com/new)
   - Name: `tp_big_data`
   - Click `Create repository`

2. **Upload files:**
   - Click `uploading an existing file`
   - Drag and drop your files
   - Commit changes

---

## 📋 Before Publishing Checklist

- [ ] **Update README:** Replace `YOUR_USERNAME` with your GitHub username
- [ ] **Add personal information:** Update the notebook with your name and email
- [ ] **Test the notebook:** Run all cells to ensure everything works
- [ ] **Create requirements.txt:** Run `pip freeze > requirements.txt`
- [ ] **Remove sensitive data:** Check for any API keys or passwords
- [ ] **Add LICENSE file:** Choose MIT, Apache 2.0, or GPL
- [ ] **Create .gitignore:** Exclude unnecessary files

---

## 🤝 Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📧 Contact

**Project Author:** [Your Name]  
**Email:** [your.email@example.com]  
**Institution:** Université Évry Paris-Saclay  
**GitHub:** [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)

---

## 🌟 Acknowledgments

- **Course Instructor:** [Instructor Name]
- **Institution:** Département TNI, Université Évry Paris-Saclay
- **Datasets:** UCI Machine Learning Repository, Scikit-learn
- **Inspiration:** Apache Spark and Scikit-learn communities

---

## 📊 Sample Visualizations

### Execution Time Comparison
![Execution Time](results/figures/execution_time_comparison.png)

### Scalability Analysis
![Scalability](results/figures/scalability_analysis.png)

### Cluster Visualization (PCA)
![Clusters](results/figures/cluster_visualizations.png)

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for Big Data Analytics

[⬆ Back to Top](#-big-data-clustering-comparative-analysis)

</div>
