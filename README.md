
# SREengine: Dynamic Simulation of Gene Expression with Knockout Modeling

**Author**: Tejaswini Jatti
**Repository**: [github.com/tejaswinijatti/SREengine](https://github.com/tejaswinijatti/SREengine)

---

## 🧬 Overview

**SREengine** is a Python-based simulation framework for modeling gene expression dynamics over time using ordinary differential equations (ODEs). The model incorporates regulatory influences (activations and repressions) between genes and allows users to simulate the effects of gene knockouts (KOs) on temporal expression profiles.

This simulation is particularly useful for exploring regulatory behavior of genes involved in lipid metabolism and oncogenic pathways.

---

## 🎯 Features

* Simulates dynamic gene expression using real RNA expression input.
* Incorporates activation and repression using Hill functions.
* Allows one or multiple gene knockouts.
* Visualizes time-course plots and shaded expression differences.
* Calculates and compares Area Under Curve (AUC) for expression dynamics.
* Exports AUC differences to Excel.

---

## 🗂️ Repository Contents

* `simulation_expression_yaxis_constant.py`: Main simulation script.
* Input CSV file (not included): Expression count matrix with gene names as row indices.
* Output: Time-series plots and Excel files summarizing AUC comparisons.

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/tejaswinijatti/SREengine.git
cd SREengine
```

2. Install required Python packages:

```bash
pip install numpy pandas matplotlib scipy
```

---

## 📥 Input Format

Provide a CSV file with:

* Rows: Gene names
* Columns: Expression samples (e.g., replicates or conditions)

Example format:

```
Gene,A,B,C
FASN,500,480,510
SCD,320,300,310
... (and so on)
```

Set the path to this file in the script under `csv_data_path`.

---

## 🚀 How to Run

Open and edit the `simulation_expression_yaxis_constant.py` file:

1. Set input file path:

```python
csv_data_path = r"path/to/your/expression_file.csv"
```

2. Define gene(s) to knock out:

```python
ko_target_genes = ["SREBF1"]  # Replace with desired KO gene(s)
```

3. Optionally, define genes to plot:

```python
selected_to_plot = ["SREBF1", "FASN", "SCD"]
```

4. Run the script:

```bash
python simulation_expression_yaxis_constant.py
```

Output:

* Time-series plots
* AUC bar chart
* Excel file with AUC changes

---

## 📊 Output Explanation

* **Plots**: Expression vs. time curves for each gene, with KO and normal conditions.
* **Shading**: Highlights which condition shows higher expression.
* **Excel**: AUC comparison including % change.

---

## 📌 Notes

* Tune the following parameters if results are flat or unresponsive:

  * `PRODUCTION_RATE_SCALING_FACTOR`
  * `STRENGTH_SCALING_FACTOR`
  * `DATA_INPUT_SCALING_FACTOR`
* These are all defined at the top of the script.

---

## 📃 License

MIT License (or specify if different).

---

## 🙋‍♀️ Author

Tejaswini Jatti

For questions, raise an issue or contact via GitHub.
