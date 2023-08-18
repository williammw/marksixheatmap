# Winning Numbers Analysis 

This repository provides mark6 history heatmap information, you can download latest mark6 history @ https://en.lottolyzer.com/history/hong-kong/mark-six

## Prerequisites

Ensure you have the following libraries installed:

- `matplotlib`
- `numpy`
- `pandas`

You can install them using `pip`:

```
pip install matplotlib numpy pandas
```

## Usage

### 1. Import Necessary Libraries

Start by importing the necessary Python libraries:

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

### 2. Extract Winning Numbers from the CSV

Load the winning numbers from your CSV file:

```python
winning_numbers_data = extract_winning_numbers("MarkSix.csv")
```

### 3. Visualize the Number Grid with Highlighted Numbers

Display a grid of numbers from 1 to 49, with winning numbers highlighted:

```python
visualize_number_grid_updated(winning_numbers_data)
```

### 4. Calculate and Visualize the Heatmap

Determine the frequency of each number's appearance and visualize it:

```python
heatmap_data = calculate_heatmap_data(winning_numbers_data)
visualize_heatmap_configurable(heatmap_data, colormap='hot', annotate=True)
```

### 5. Histogram of Winning Numbers

Show the distribution of winning numbers:

```python
plot_histogram(winning_numbers_data)
```

### 6. Basic Statistics

Obtain insights such as the mean, median, and mode of the winning numbers:

```python
mean_val, median_val, mode_val = calculate_statistics(winning_numbers_data)
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val)
```

## Customizations

You can adjust various parameters in the functions to customize visualizations. For example, in `visualize_heatmap_configurable`, you can change the `colormap` parameter to use different color schemes and set `annotate` to `False` if you don't want annotations.

---

Copy the above markdown code and place it in a `README.md` file in your repository or project directory. This will provide a clear guide for anyone looking to use the tools and functions you've developed.