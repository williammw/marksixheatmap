
# %%
# Only import necessary libraries once
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Updated function to visualize the grid with highlighted numbers


def extract_winning_numbers(file_path, num_records=10):
    """
    Extract the winning numbers from the CSV file.

    Parameters:
    - file_path: Path to the CSV file.
    - num_records: Number of records to extract (default is 10).

    Returns:
    - DataFrame with the extracted winning numbers.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Extract the desired columns and limit the number of records
    extracted_data = df[['Winning Number 1', '2',
                        '3', '4', '5', '6']].head(num_records)

    return extracted_data


# Extract the first 10 winning numbers for testing
winning_numbers_data = extract_winning_numbers("MarkSix.csv")
winning_numbers_data


def visualize_highlighted_numbers_updated(winning_numbers_data, size=(12, 12)):
    fig, ax = plt.subplots(figsize=size)

    # Create a matrix to count the number of repetitions for each number
    heatmap_data = np.zeros((7, 7))

    # Iterate over the winning numbers and update the heatmap data
    for _, row in winning_numbers_data.iterrows():
        for num in row:
            x = (num-1) % 7
            y = 6 - (num-1) // 7
            heatmap_data[y, x] += 1

    # Define the numbers to be displayed
    numbers = np.arange(1, 50).reshape(7, 7)

    # Hide the axes
    ax.axis('off')

    # Adjust the axis limits to ensure numbers are centered
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)

    # Size of each square
    square_size = 1.0

    # Display each number in its position and draw rectangles around them
    for i in range(7):
        for j in range(7):
            ax.text(j, 6-i, str(numbers[i, j]),
                    ha='center', va='center', fontsize=20,
                    color='red' if heatmap_data[6-i, j] > 0 else 'black')

            # Draw rectangle with overlapping borders
            rect = plt.Rectangle((j-0.5, 6.5-i), square_size, -square_size,
                                 edgecolor='black', facecolor='none', lw=1.5)
            ax.add_patch(rect)

    plt.show()

    return heatmap_data

# Placeholder for the remaining steps
# The subsequent steps will be implemented in the following code cells


def visualize_heatmap_enhanced(heatmap_data, size=(12, 12), colormap='viridis'):
    fig, ax = plt.subplots(figsize=size)

    # Display the heatmap with dynamic color scaling
    max_val = np.max(heatmap_data)
    cax = ax.matshow(heatmap_data, cmap=colormap, vmin=0, vmax=max_val)

    # Define the numbers to be displayed
    numbers = np.arange(1, 50).reshape(7, 7)

    # Hide the axes
    ax.axis('off')

    # Display each number on the heatmap and annotate with actual count
    for i in range(7):
        for j in range(7):
            num = numbers[i, j]
            count = int(heatmap_data[i, j])
            ax.text(j, i, str(num),
                    ha='center', va='center', fontsize=16,
                    color='white' if count > max_val * 0.5 else 'black')
            if count:
                ax.text(j, i+0.3, str(count),
                        ha='center', va='center', fontsize=10,
                        color='white' if count > max_val * 0.5 else 'black')

    # Display the colorbar to show the scale
    plt.colorbar(cax, orientation='vertical', shrink=0.75)

    plt.show()

# Placeholder for the remaining steps
# The subsequent steps will be implemented in the following code cells


def plot_histogram(winning_numbers_data, size=(10, 6)):
    # Flatten the data to a 1D array
    flat_data = winning_numbers_data.values.flatten()

    fig, ax = plt.subplots(figsize=size)

    # Plot histogram
    ax.hist(flat_data, bins=np.arange(0.5, 51.5, 1),
            edgecolor='black', alpha=0.7, color='skyblue')

    ax.set_title("Frequency Distribution of Winning Numbers")
    ax.set_xlabel("Number")
    ax.set_ylabel("Frequency")
    ax.set_xticks(np.arange(1, 51))

    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

# Placeholder for the remaining steps
# The subsequent steps will be implemented in the following code cells


def calculate_statistics(winning_numbers_data):
    # Flatten the data to a 1D array
    flat_data = winning_numbers_data.values.flatten()

    mean_val = np.mean(flat_data)
    median_val = np.median(flat_data)
    (values, counts) = np.unique(flat_data, return_counts=True)
    mode_val = values[np.argmax(counts)]

    return mean_val, median_val, mode_val


mean_val, median_val, mode_val = calculate_statistics(winning_numbers_data)

mean_val, median_val, mode_val


def visualize_number_grid_updated(winning_numbers_data=None, size=(12, 12)):
    fig, ax = plt.subplots(figsize=size)

    # Create a matrix to count the number of repetitions for each number
    heatmap_data = np.zeros(
        (7, 7)) if winning_numbers_data is None else calculate_heatmap_data(winning_numbers_data)

    # Define the numbers to be displayed
    numbers = np.arange(1, 50).reshape(7, 7)

    # Hide the axes
    ax.axis('off')

    # Adjust the axis limits to ensure numbers are centered
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-0.5, 6.5)

    # Size of each square
    square_size = 1.0

    # Display each number in its position and draw rectangles around them
    for i in range(7):
        for j in range(7):
            ax.text(j, 6-i, str(numbers[i, j]),
                    ha='center', va='center', fontsize=20,
                    color='red' if heatmap_data[6-i, j] > 0 else 'black')

            # Draw rectangle with overlapping borders
            rect = plt.Rectangle((j-0.5, 6.5-i), square_size, -square_size,
                                 edgecolor='black', facecolor='none', lw=1.5)
            ax.add_patch(rect)

    plt.show()


def calculate_heatmap_data(winning_numbers_data):
    heatmap_data = np.zeros((7, 7))
    for _, row in winning_numbers_data.iterrows():
        for num in row:
            x = (num-1) % 7
            y = 6 - (num-1) // 7
            heatmap_data[y, x] += 1
    return heatmap_data

# Placeholder for the remaining steps
# The subsequent steps will be implemented in the following code cells


def visualize_heatmap_configurable(heatmap_data=None, size=(12, 12), colormap='magma', annotate=True):
    fig, ax = plt.subplots(figsize=size)

    # If heatmap_data is None, create an empty heatmap
    if heatmap_data is None:
        heatmap_data = np.zeros((7, 7))

    # Display the heatmap with dynamic color scaling
    max_val = np.max(heatmap_data)
    cax = ax.matshow(heatmap_data, cmap=colormap, vmin=0, vmax=max_val)

    # Define the numbers to be displayed
    numbers = np.arange(1, 50).reshape(7, 7)

    # Hide the axes
    ax.axis('off')

    # Display each number on the heatmap and optionally annotate with actual count
    for i in range(7):
        for j in range(7):
            num = numbers[i, j]
            count = int(heatmap_data[i, j])
            ax.text(j, i, str(num),
                    ha='center', va='center', fontsize=16,
                    color='white' if count > max_val * 0.5 else 'black')

            if annotate and count:
                ax.text(j, i+0.3, str(count),
                        ha='center', va='center', fontsize=10,
                        color='white' if count > max_val * 0.5 else 'black')

    # Display the colorbar to show the scale
    plt.colorbar(cax, orientation='vertical', shrink=0.75)

    plt.show()


winning_numbers_data = extract_winning_numbers("MarkSix.csv", 30)
visualize_number_grid_updated(winning_numbers_data)
visualize_number_grid_updated(winning_numbers_data)
heatmap_data = calculate_heatmap_data(winning_numbers_data)
visualize_heatmap_configurable(heatmap_data, colormap='hot', annotate=True)
plot_histogram(winning_numbers_data)
mean_val, median_val, mode_val = calculate_statistics(winning_numbers_data)
print("Mean:", mean_val)
print("Median:", median_val)
print("Mode:", mode_val)

# %%
