# Coursera Discrete Optimization – My Answers  

My solutions to the weekly assignments of the **Discrete Optimization** course from the **University of Melbourne** (2023, Coursera).  

---

## 📊 Grades  

| Week | Assignment            | Score   |
|------|------------------------|---------|
| 1    | Any integer            | 10 / 10 |
| 2    | Knapsack               | 60 / 60 |
| 3    | Graph coloring         | 45 / 60 |
| 4    | Traveling salesman     | 42 / 60 |
| 5    | Facility location      | 58 / 80 |
| 6    | Vehicle routing        | 48 / 60 |


## Requirements

This repository contains solutions and code for discrete optimization problems, implemented primarily in Python, with some Java utilities. Below are the main dependencies required to run the code.

---

### Python (for most scripts)

- Python 3.x
- numpy
- pandas
- matplotlib
- pyomo
- scikit-learn
- bitarray

You can install all Python dependencies with:

```bash
pip install numpy pandas matplotlib pyomo scikit-learn bitarray
```

---

### Jupyter Notebook (optional)

If you wish to run or explore the `.ipynb` notebook files, you will also need:

- jupyter

Install Jupyter with:

```bash
pip install jupyter
```

---

### Java (for specific knapsack solutions in `/Week2/Knapsack/discardJava`)

Some solutions, especially in the `discardJava` folder, require Java:

- Java 8 or higher

---

### Standard Libraries

The code also uses several Python standard libraries, which come pre-installed with Python:

- collections
- json
- time
- os
- math
- sys
- random
- subprocess
- operator

---

## Notes

- Not all dependencies are used in every script. Check the import statements at the top of each script for more details.
- For scientific computing and optimization, `numpy`, `pandas`, `pyomo`, and `scikit-learn` are the main third-party libraries used.
- Some solutions use visualization (`matplotlib`) or bit manipulation (`bitarray`).
- Java solutions are stand-alone and should be compiled/run separately.
- If you encounter missing packages, please refer to the import statements in the relevant script.

---
