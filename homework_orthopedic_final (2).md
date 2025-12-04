# Homework: Pandas, NumPy & Visualization
## Total Points: 10
## Dataset: Biomechanical Features of Orthopedic Patients

---

## Part 1: Code Refactoring (4 points)

### Task: Clean Up This Messy Code

A colleague wrote this code to analyze gene expression data, but it's difficult to understand and maintain. **Refactor it following clean code principles.**

```python
import pandas as pd

def a(f):
    d = pd.read_csv(f)
    r = []
    for i in range(len(d)):
        if d.iloc[i]['age'] >= 50 and d.iloc[i]['cancer_type'] == 'Lung':
            if d.iloc[i]['final_tumor_size'] < d.iloc[i]['baseline_tumor_size']:
                r.append(d.iloc[i]['patient_id'])
    
    x = 0
    y = 0
    for i in range(len(d)):
        if d.iloc[i]['treatment'] == 'Treatment_A':
            x += d.iloc[i]['survival_months']
        if d.iloc[i]['treatment'] == 'Control':
            y += d.iloc[i]['survival_months']
    
    cnt1 = len([i for i in range(len(d)) if d.iloc[i]['treatment'] == 'Treatment_A'])
    cnt2 = len([i for i in range(len(d)) if d.iloc[i]['treatment'] == 'Control'])
    
    print("Avg survival Treatment A:", x/cnt1 if cnt1 > 0 else 0)
    print("Avg survival Control:", y/cnt2 if cnt2 > 0 else 0)
    print("Responsive patients:", len(r))
    
    return r

result = a('clinical_trial_patients.csv')
```

### Requirements for refactoring (4 points total):

**1. Descriptive names (1 point):**
- Replace cryptic variable names (a, d, r, x, y, cnt1, cnt2, f)
- Use meaningful function names
- Variables should explain their purpose

**2. Remove magic numbers (0.5 points):**
- Define constants for the age threshold (50)
- Use named constants at the top of your code
- Constants should be UPPERCASE

**3. Split into functions (1.5 points):**
- One function should load data
- One function should find responsive patients
- One function should calculate survival statistics
- Main function coordinates these

**4. Add documentation (0.5 points):**
- Docstrings for all functions
- Explain parameters and return values

**5. Better pandas usage (0.5 points):**
- Replace `.iloc` loops with pandas filtering
- Use `.groupby()` where appropriate
- Avoid iterating over rows when possible

### Submission for Part 1:
- Submit file: `refactored_code.py`
- Code should be clean, readable, and follow PEP 8

---

## Part 2: Data Analysis with Orthopedic Dataset (6 points)

### About the Dataset

You will analyze biomechanical features of orthopedic patients from the UCI Machine Learning Repository.

**Dataset:** 310 patients with spine measurements

**Columns:**
- `pelvic_incidence` - Pelvic incidence angle (degrees)
- `pelvic_tilt` - Pelvic tilt angle (degrees)  
- `lumbar_lordosis_angle` - Lumbar lordosis angle (degrees)
- `sacral_slope` - Sacral slope angle (degrees)
- `pelvic_radius` - Pelvic radius (mm)
- `degree_spondylolisthesis` - Degree of spondylolisthesis (percentage)
- `class` - Diagnosis: **Normal**, **Hernia**, or **Spondylolisthesis**

### Download the dataset:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load from URL
url = 'https://raw.githubusercontent.com/reisanar/datasets/master/column_3C_weka.csv'
patients = pd.read_csv(url)

# OR download file and load locally
patients = pd.read_csv('column_3C_weka.csv')
```

---

### Question 1: Pandas - Data Exploration & Filtering (1 point)

**Task:** Explore the dataset and filter patients with severe conditions.

**Part A (0.4 pts):** Load the data and display:
- First 5 rows
- Dataset shape (rows, columns)
- Column names and data types

**Part B (0.6 pts):** Find all patients with SEVERE biomechanical abnormalities:
- `degree_spondylolisthesis` > 30 OR
- `pelvic_incidence` > 70
- Display how many patients match and show their class distribution

```python
import pandas as pd

# Part A: Exploration
patients = pd.read_csv('column_3C_weka.csv')

print("First 5 patients:")
# Your code

print(f"\nDataset shape: ...")
print(f"Columns: ...")

# Part B: Filter severe cases
severe_cases = patients[
    # Your conditions
]

print(f"\nSevere cases found: {len(severe_cases)}")
print("\nClass distribution of severe cases:")
# Show value_counts
```

**What to submit:** Code that loads, explores, and filters data

---

### Question 2: Pandas - Group Analysis (1 point)

**Task:** Compare biomechanical features across the three diagnosis groups.

**Requirements:**

a) For EACH diagnosis class, calculate: (0.7 pts)
   - Mean of `degree_spondylolisthesis`
   - Mean of `pelvic_incidence`
   - Standard deviation of `lumbar_lordosis_angle`
   - Count of patients

b) Which diagnosis has the highest average `degree_spondylolisthesis`? (0.3 pts)

```python
# a) Group statistics
diagnosis_stats = patients.groupby('class').agg({
    # Your aggregations here
})

print("Statistics by Diagnosis:")
print(diagnosis_stats.round(2))

# b) Highest spondylolisthesis
highest_diagnosis = 

print(f"\nDiagnosis with highest spondylolisthesis: {highest_diagnosis}")
```

**Expected output format:**
```
                      degree_spondylolisthesis  pelvic_incidence  lumbar_lordosis_angle  count
class                                                                                           
Hernia                                   23.45             55.78                  15.23     60
Normal                                   13.37             49.65                  12.45    100
Spondylolisthesis                        37.89             63.21                  18.67    150
```

**What to submit:** Complete group analysis with aggregations

---

### Question 3: NumPy - Statistical Analysis (1 point)

**Task:** Use NumPy to perform numerical calculations on biomechanical features.

**Requirements:**

a) Convert `pelvic_incidence` column to NumPy array and calculate: (0.5 pts)
   - Mean
   - Median  
   - Standard deviation
   - 25th and 75th percentile (use `np.percentile`)

b) Standardize (z-score normalize) the `degree_spondylolisthesis` values: (0.5 pts)
   - Formula: `z = (x - mean) / std`
   - Use NumPy operations (no loops!)
   - Add the standardized values as a new column `spondylo_zscore`
   - Count how many patients have z-score > 2 (outliers)

```python
import numpy as np

# a) NumPy statistics
pelvic_incidence_array = patients['pelvic_incidence'].to_numpy()

mean_pi = np.mean(pelvic_incidence_array)
median_pi = np.median(pelvic_incidence_array)
std_pi = np.std(pelvic_incidence_array)
p25 = np.percentile(pelvic_incidence_array, 25)
p75 = np.percentile(pelvic_incidence_array, 75)

print("Pelvic Incidence Statistics:")
print(f"Mean: {mean_pi:.2f}")
print(f"Median: {median_pi:.2f}")
print(f"Std Dev: {std_pi:.2f}")
print(f"25th percentile: {p25:.2f}")
print(f"75th percentile: {p75:.2f}")

# b) Z-score normalization using NumPy
spondylo_array = patients['degree_spondylolisthesis'].to_numpy()

# Calculate z-scores (vectorized operation!)
spondylo_mean = np.mean(spondylo_array)
spondylo_std = np.std(spondylo_array)
spondylo_zscore = (spondylo_array - spondylo_mean) / spondylo_std

# Add to dataframe
patients['spondylo_zscore'] = spondylo_zscore

# Count outliers (|z-score| > 2)
outliers = np.abs(spondylo_zscore) > 2
num_outliers = np.sum(outliers)

print(f"\nNumber of outliers (|z-score| > 2): {num_outliers}")
print(f"Percentage of outliers: {(num_outliers/len(patients)*100):.1f}%")
```

**What to submit:** NumPy calculations with z-score normalization

---

### Question 4: Matplotlib - Visualization (1.5 points)

**Task:** Create visualizations to understand the data better.

**Requirements:**

Create a figure with **3 subplots** (use `plt.subplots()`):

**Plot 1 (0.5 pts):** Histogram of `degree_spondylolisthesis`
- 30 bins
- Add vertical line for mean value
- Label axes and add title

**Plot 2 (0.5 pts):** Box plot comparing `pelvic_incidence` across the three diagnosis classes
- Different colors for each class
- Label axes and add title
- Rotate x-axis labels if needed

**Plot 3 (0.5 pts):** Scatter plot of `pelvic_incidence` vs `degree_spondylolisthesis`
- Color points by diagnosis class (3 different colors)
- Add legend
- Label axes and add title

```python
import matplotlib.pyplot as plt

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Histogram
axes[0].hist(patients['degree_spondylolisthesis'], bins=30, 
             color='skyblue', edgecolor='black', alpha=0.7)
mean_spondylo = patients['degree_spondylolisthesis'].mean()
axes[0].axvline(mean_spondylo, color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {mean_spondylo:.1f}')
axes[0].set_xlabel('Degree of Spondylolisthesis', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Spondylolisthesis', fontsize=14)
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Plot 2: Box plot
# Prepare data for box plot
diagnosis_groups = [
    patients[patients['class'] == 'Normal']['pelvic_incidence'],
    patients[patients['class'] == 'Hernia']['pelvic_incidence'],
    patients[patients['class'] == 'Spondylolisthesis']['pelvic_incidence']
]

bp = axes[1].boxplot(diagnosis_groups, 
                      labels=['Normal', 'Hernia', 'Spondylolisthesis'],
                      patch_artist=True)
# Color the boxes
colors = ['lightgreen', 'orange', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

axes[1].set_ylabel('Pelvic Incidence (degrees)', fontsize=12)
axes[1].set_xlabel('Diagnosis', fontsize=12)
axes[1].set_title('Pelvic Incidence by Diagnosis', fontsize=14)
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

# Plot 3: Scatter plot
# Color by class
colors_map = {'Normal': 'green', 'Hernia': 'orange', 'Spondylolisthesis': 'red'}
for diagnosis in patients['class'].unique():
    diagnosis_data = patients[patients['class'] == diagnosis]
    axes[2].scatter(diagnosis_data['pelvic_incidence'], 
                    diagnosis_data['degree_spondylolisthesis'],
                    c=colors_map[diagnosis], 
                    label=diagnosis, 
                    alpha=0.6, 
                    s=50,
                    edgecolors='black',
                    linewidth=0.5)

axes[2].set_xlabel('Pelvic Incidence (degrees)', fontsize=12)
axes[2].set_ylabel('Degree of Spondylolisthesis', fontsize=12)
axes[2].set_title('Pelvic Incidence vs Spondylolisthesis', fontsize=14)
axes[2].legend(title='Diagnosis')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('orthopedic_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

**What to submit:** Code creating all 3 plots + saved figure file

---

### Question 5: Pandas - Correlation & Complex Query (0.75 points)

**Task:** Analyze relationships between features and find specific patient groups.

**Part A (0.4 pts):** Calculate correlation matrix and find the two features with the STRONGEST positive correlation (excluding self-correlation).

**Part B (0.35 pts):** Find patients who meet ALL these criteria:
- `pelvic_incidence` between 50 and 70 (inclusive)
- `degree_spondylolisthesis` > 20
- `class` is "Spondylolisthesis"
- Show count and percentage of total patients

```python
# Part A: Correlation
numeric_data = patients.select_dtypes(include=['number'])
correlation_matrix = numeric_data.corr()

print("Correlation Matrix:")
print(correlation_matrix.round(3))

# Find highest correlation (excluding diagonal)
# Hint: use .unstack() and filter out 1.0 values


# Part B: Complex filtering
specific_patients = patients[
    # Your multiple conditions
]

print(f"\nPatients matching criteria: {len(specific_patients)}")
print(f"Percentage of total: {(len(specific_patients)/len(patients)*100):.1f}%")
```

**What to submit:** Correlation analysis + complex filtering

---

### Question 6: Pandas - Summary Report Creation (0.75 points)

**Task:** Create a comprehensive summary comparing Normal vs Abnormal patients.

**Requirements:**

a) Create a new column `abnormal` that is `True` if class is "Hernia" or "Spondylolisthesis", `False` if "Normal" (0.25 pts)

b) Compare Normal vs Abnormal groups by calculating: (0.5 pts)
   - Count of patients
   - Mean of ALL numeric features
   - Present as a clean comparison table

```python
# a) Create abnormal flag
patients['abnormal'] = patients['class'].isin(['Hernia', 'Spondylolisthesis'])

# OR
patients['abnormal'] = patients['class'] != 'Normal'

print(f"Normal patients: {(~patients['abnormal']).sum()}")
print(f"Abnormal patients: {patients['abnormal'].sum()}")

# b) Compare groups
comparison = patients.groupby('abnormal').agg({
    'pelvic_incidence': 'mean',
    'pelvic_tilt': 'mean',
    'lumbar_lordosis_angle': 'mean',
    'sacral_slope': 'mean',
    'pelvic_radius': 'mean',
    'degree_spondylolisthesis': 'mean'
})

comparison['patient_count'] = patients.groupby('abnormal').size()

print("\nNormal vs Abnormal Comparison:")
print(comparison.round(2))

# Which features differ most?
differences = comparison.loc[True] - comparison.loc[False]
biggest_diff = differences.drop('patient_count').abs().idxmax()

print(f"\nFeature with biggest difference: {biggest_diff}")
print(f"Difference: {differences[biggest_diff]:.2f}")
```

**Expected output:**
```
Normal vs Abnormal Comparison:
           pelvic_incidence  pelvic_tilt  ...  degree_spondylolisthesis  patient_count
abnormal                                  ...                                          
False                 49.65        15.23  ...                     13.37            100
True                  67.89        21.45  ...                     32.56            210

Feature with biggest difference: degree_spondylolisthesis
Difference: 19.19
```

**What to submit:** Complete comparison with difference analysis

---

## Submission Guidelines

### What to submit:

1. **Part 1:** `refactored_code.py` - Your cleaned-up code
2. **Part 2:** `orthopedic_analysis.py` or `orthopedic_analysis.ipynb` 
   - All 6 questions answered
   - Figure saved as `orthopedic_analysis.png`

### File naming:
- `lastname_firstname_homework.py` or `.ipynb`
- Include the PNG file from Question 4

### Requirements:
- ✅ Code must run without errors
- ✅ Include comments explaining logic
- ✅ Import all necessary libraries
- ✅ Follow PEP 8 style

### Testing your code:

```python
# Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
patients = pd.read_csv('column_3C_weka.csv')

# Verify
print(f"Shape: {patients.shape}")  # Should be (310, 7)
print(f"Columns: {list(patients.columns)}")
```

---

## Grading Rubric

### Part 1: Code Refactoring (4 points)

| Criteria | Points |
|----------|--------|
| Descriptive names | 1.0 |
| Constants (UPPERCASE) | 0.5 |
| Function separation | 1.5 |
| Documentation (docstrings) | 0.5 |
| Pandas best practices | 0.5 |

### Part 2: Data Analysis (6 points)

| Question | Points | Focus |
|----------|--------|-------|
| Q1: Exploration & Filtering | 1.00 | Pandas basics |
| Q2: Group Analysis | 1.00 | Pandas groupby |
| Q3: Statistical Analysis | 1.00 | NumPy operations |
| Q4: Visualization | 1.50 | Matplotlib (3 plots) |
| Q5: Correlation & Query | 0.75 | Pandas advanced |
| Q6: Summary Report | 0.75 | Pandas synthesis |

**Total: 10 points**

---

## Tips for Success

### NumPy Tips (Q3):
- ✅ Convert pandas Series to array: `.to_numpy()` or `.values`
- ✅ Use vectorized operations (no loops!)
- ✅ `np.mean()`, `np.std()`, `np.percentile()`
- ✅ Broadcasting: operations on entire arrays

### Matplotlib Tips (Q4):
- ✅ Create subplots: `fig, axes = plt.subplots(1, 3, figsize=(18, 5))`
- ✅ Access individual axes: `axes[0]`, `axes[1]`, `axes[2]`
- ✅ Always label axes and add titles
- ✅ Use `plt.tight_layout()` for clean spacing
- ✅ Save with high DPI: `plt.savefig('file.png', dpi=300)`

### Pandas Tips (Q1, Q2, Q5, Q6):
- ✅ Boolean filtering: `df[(condition1) & (condition2)]`
- ✅ GroupBy aggregations: `.groupby().agg({})`
- ✅ Value counts: `.value_counts()`
- ✅ Correlation: `.corr()`

---

## Common Mistakes to Avoid

### NumPy:
❌ Using loops instead of vectorized operations  
❌ Forgetting to convert to numpy array  
❌ Wrong formula for z-score  
✅ Use numpy functions on entire arrays  

### Matplotlib:
❌ Not labeling axes  
❌ Plots too small or unclear  
❌ Forgetting `plt.show()` or `plt.savefig()`  
✅ Clear labels, titles, legends  

### Pandas:
❌ Using `and`/`or` instead of `&`/`|`  
❌ Missing parentheses in conditions  
❌ Wrong aggregation functions  
✅ Check output matches expected format  

---

## Dataset Download

**Option 1 - Direct URL:**
```python
url = 'https://raw.githubusercontent.com/reisanar/datasets/master/column_3C_weka.csv'
patients = pd.read_csv(url)
```

**Option 2 - Kaggle:**
1. Go to: https://www.kaggle.com/datasets/uciml/biomechanical-features-of-orthopedic-patients
2. Download `column_3C_weka.csv`
3. Place in your working directory

---

## Example Code Structure

```python
"""
Orthopedic Patient Analysis
Student: [Your Name]
Date: [Date]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==================== Question 1 ====================
print("=" * 50)
print("QUESTION 1: Data Exploration & Filtering")
print("=" * 50)

# Load data
patients = pd.read_csv('column_3C_weka.csv')

# Part A
# ... your code

# Part B
# ... your code

# ==================== Question 2 ====================
print("\n" + "=" * 50)
print("QUESTION 2: Group Analysis")
print("=" * 50)

# ... your code

# Continue for all questions...
```

---

## Academic Integrity

**Allowed:**
- ✅ Use pandas/numpy/matplotlib documentation
- ✅ Search for specific syntax
- ✅ Discuss concepts with classmates
- ✅ Ask instructor for help

**Not Allowed:**
- ❌ Copy code from others
- ❌ Share your solutions
- ❌ Use complete solutions from internet
- ❌ Have AI write all your code

**Citation:** If you use external resources, cite them:
```python
# Reference: NumPy percentile documentation
# https://numpy.org/doc/stable/reference/generated/numpy.percentile.html
```

---

## Estimated Time

- **Part 1 (Refactoring):** 1 hour
- **Q1-Q2 (Pandas basics):** 45 min
- **Q3 (NumPy):** 30 min
- **Q4 (Matplotlib):** 45 min
- **Q5-Q6 (Advanced pandas):** 1 hour
- **Testing & debugging:** 30 min

**Total: 4-5 hours**

---

## Need Help?

**Before asking:**
1. Read error messages carefully
2. Check documentation
3. Verify your data loaded correctly
4. Test with small examples

**When stuck:**
- Office hours: [Your hours]
- Email: [Your email]
- Include: question number, error message, what you've tried

---

## Good Luck! 🦴📊

**Remember:**
- Start early
- Test each question before moving on
- Comment your code
- Make your plots clear and labeled

**The goal is learning NumPy, Pandas, and Matplotlib - not just getting points!**

---

**Due Date:** [Add deadline]
