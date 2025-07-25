import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import seaborn as sns
import os
from math import *

# The function to transpose the content of the file
def transpose_oriAA_result(fname, fnameout):
   # Read the CSV file
    file_path = fname
    df = pd.read_csv(file_path)
    
    # Transpose the DataFrame
    transposed_df = df.transpose()
    
    # Add a new header for the columns of the transposed DataFrame
    new_header = ['Col_' + str(i) for i in range(1, len(transposed_df.columns) + 1)]
    transposed_df.columns = new_header

    # Write the transposed DataFrame to a new CSV file
    output_file_path = fnameout
    transposed_df.to_csv(output_file_path, index=False)

# The function to add middle value (probably 0) to the file to make visualization of 24-2 visual field test easier to display. This one for right visualization
def add_zeros_for24_2(fname, fnameout, val):
    df1 = pd.read_csv(fname)
    nol_value = val
    blind_spot = 0
    df1.insert (0, "n1", nol_value)
    df1.insert (1, "n2", nol_value)
    df1.insert (2, "n3", nol_value)
    df1.insert (7, "n4", nol_value)
    df1.insert (8, "n5", nol_value)
    df1.insert (9, "n6", nol_value)
    df1.insert (10, "n7", nol_value)
    df1.insert (17, "n8", nol_value)
    df1.insert (18, "n9", nol_value)
    df1.insert (34, "n10", blind_spot)
    df1.insert (43, "n11", blind_spot)
    df1.insert (45, "n12", nol_value)
    df1.insert (54, "n13", nol_value)
    df1.insert (55, "n14", nol_value)
    df1.insert (62, "n15", nol_value)
    df1.insert (63, "n16", nol_value)
    df1.insert (64, "n17", nol_value)
    df1.insert (65, "n18", nol_value)
    df1.insert (70, "n19", nol_value)
    df1.insert (71, "n20", nol_value)
    df1.insert (72, "n21", nol_value)
    df1.insert (73, "n22", nol_value)
    df1.insert (74, "n23", nol_value)
    df1.insert (75, "n24", nol_value)
    df1.insert (76, "n25", nol_value)
    df1.insert (77, "n26", nol_value)
    df1.insert (78, "n27", nol_value)
    df1.insert (79, "n28", nol_value)
    df1.insert (80, "n29", nol_value)
    
    df1.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
                 '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80']
    
    df1.to_excel(fnameout)
    
# The function to add middle value (probably 0) to the file to make visualization of 24-2 visual field test easier to display. This one for left visualization.
def add_zeros_for24_2_left(fname, fnameout, val):
    df1 = pd.read_csv(fname)
    nol_value = val
    df1.insert (0, "n1", nol_value)
    df1.insert (1, "n2", nol_value)
    df1.insert (6, "n3", nol_value)
    df1.insert (7, "n4", nol_value)
    df1.insert (8, "n5", nol_value)
    df1.insert (9, "n6", nol_value)
    df1.insert (16, "n7", nol_value)
    df1.insert (17, "n8", nol_value)
    df1.insert (26, "n9", nol_value)
    df1.insert (28, "n10", nol_value)
    df1.insert (37, "n11", nol_value)
    df1.insert (53, "n12", nol_value)
    df1.insert (54, "n13", nol_value)
    df1.insert (61, "n14", nol_value)
    df1.insert (62, "n15", nol_value)
    df1.insert (63, "n16", nol_value)
    df1.insert (64, "n17", nol_value)
    df1.insert (69, "n18", nol_value)
    df1.insert (70, "n19", nol_value)
    df1.insert (71, "n20", nol_value)
    df1.insert (72, "n21", nol_value)
    df1.insert (73, "n22", nol_value)
    df1.insert (74, "n23", nol_value)
    df1.insert (75, "n24", nol_value)
    df1.insert (76, "n25", nol_value)
    df1.insert (77, "n26", nol_value)
    df1.insert (78, "n27", nol_value)
    df1.insert (79, "n28", nol_value)
    df1.insert (80, "n29", nol_value)
    
    df1.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
                 '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80']
    
    df1.to_excel(fnameout)

# The function to add middle value (probably 0) to the file to make visualization of 30-2 visual field test easier to display 
def add_zeros_for30_2(fname, fnameout, val):
    df1 = pd.read_csv(fname)
    nol_value = val
    df1.insert (0, "n1", nol_value)
    df1.insert (1, "n2", nol_value)
    df1.insert (2, "n3", nol_value)
    df1.insert (7, "n4", nol_value)
    df1.insert (8, "n5", nol_value)
    df1.insert (9, "n6", nol_value)
    df1.insert (10, "n7", nol_value)
    df1.insert (11, "n8", nol_value)
    df1.insert (18, "n9", nol_value)
    df1.insert (19, "n10", nol_value)
    df1.insert (20, "n11", nol_value)
    df1.insert (29, "n12", nol_value)
    df1.insert (47, "n13", nol_value)
    df1.insert (57, "n14", nol_value)
    df1.insert (70, "n15", nol_value)
    df1.insert (79, "n16", nol_value)
    df1.insert (80, "n17", nol_value)
    df1.insert (81, "n18", nol_value)
    df1.insert (88, "n19", nol_value)
    df1.insert (89, "n20", nol_value)
    df1.insert (90, "n21", nol_value)
    df1.insert (91, "n22", nol_value)
    df1.insert (92, "n23", nol_value)
    df1.insert (97, "n24", nol_value)
    df1.insert (98, "n25", nol_value)
    df1.insert (99, "n26", nol_value)
    
    df1.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
                 '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
                 '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    
    df1.to_excel(fnameout)

# The function to add middle value (probably 0) to the file to make visualization of 10-2 visual field test easier to display 
def add_zeros_for10_2(fname, fnameout, val):
    df1 = pd.read_csv(fname)
    nol_value = val
    df1.insert (0, "n1", nol_value)
    df1.insert (1, "n2", nol_value)
    df1.insert (2, "n3", nol_value)
    df1.insert (3, "n4", nol_value)
    df1.insert (6, "n5", nol_value)
    df1.insert (7, "n6", nol_value)
    df1.insert (8, "n7", nol_value)
    df1.insert (9, "n8", nol_value)
    df1.insert (10, "n9", nol_value)
    df1.insert (11, "n10", nol_value)
    df1.insert (18, "n11", nol_value)
    df1.insert (19, "n12", nol_value)
    df1.insert (20, "n13", nol_value)
    df1.insert (29, "n14", nol_value)
    df1.insert (30, "n15", nol_value)
    df1.insert (39, "n16", nol_value)
    df1.insert (60, "n17", nol_value)
    df1.insert (69, "n18", nol_value)
    df1.insert (70, "n19", nol_value)
    df1.insert (79, "n20", nol_value)
    df1.insert (80, "n21", nol_value)
    df1.insert (81, "n22", nol_value)
    df1.insert (88, "n23", nol_value)
    df1.insert (89, "n24", nol_value)
    df1.insert (90, "n25", nol_value)
    df1.insert (91, "n26", nol_value)
    df1.insert (92, "n27", nol_value)
    df1.insert (93, "n28", nol_value)
    df1.insert (96, "n29", nol_value)
    df1.insert (97, "n30", nol_value)
    df1.insert (98, "n31", nol_value)
    df1.insert (99, "n32", nol_value)
    
    df1.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', 
                 '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45',
                 '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67',
                 '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89',
                 '90', '91', '92', '93', '94', '95', '96', '97', '98', '99']
    
    df1.to_excel(fnameout)


def plot_result(fname, dat, min=-30, center=0, max=30, r=4, c=5, td=9, fwidth=15, fheight=10, text=False, bsgrey=False):
    # Define custom red-white-blue colormap
    colors = ["#ff0000", "#ffffff", "#a9def9"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("CustomRedWhiteBlue", colors)
    cmap = custom_cmap

    # Create grey-only colormap
    grey_cmap = ListedColormap(["grey"])

    # Define normalization
    norm = mcolors.TwoSlopeNorm(vmin=min, vcenter=center, vmax=max)

    # Create subplot grid
    fig, axes = plt.subplots(r, c, figsize=(fwidth, fheight))
    x = 0

    for i in range(r):
        for j in range(c):
            if x >= dat.shape[0]:
                data = np.full((td, td), center)
            else:
                data = dat[x].reshape((td, td))

            # Create masks
            mask_main = np.full_like(data, False, dtype=bool)
            mask_grey = np.full_like(data, True, dtype=bool)

            # Blind spot positions: (y=3,x=7) → row=3, col=7 and (y=4,x=7) → row=4, col=7
            blind_spots = [(3, 7), (4, 7)]
            for r_b, c_b in blind_spots:
                mask_main[r_b, c_b] = True
                mask_grey[r_b, c_b] = False

            # Plot regular data
            sns.heatmap(data, ax=axes[i, j], annot=text, annot_kws={"size": 6},
                        cmap=cmap, norm=norm, linewidths=0.5, cbar=False, mask=mask_main)

            if bsgrey == True:
                # Plot grey over blind spots
                sns.heatmap(data, ax=axes[i, j], annot=text, annot_kws={"size": 6},
                            cmap=grey_cmap, norm=norm, linewidths=0.5, cbar=False, mask=mask_grey)

            axes[i, j].set_title(f"Archetype {x+1}")
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            x += 1

    fig.suptitle("Archetypes for Visual Field", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(fname + '.png')
    plt.show()
    

# Function to visualize the result in r rows, c columns. 
def plot_result_onebyone(fname, dat, min=-30, center=0, max=30, td=9, fwidth=4, fheight=3, annotasi=False, bsgrey=False):
    # td = 9 for display in 9x9 for 24-2. For 30-2 and 10-2, it will be td=10 (10x10)
    # fwidth and fheight for 24-2 and 30-2 is 15 and 10. But for 10-2 it is 12 and 7
    # Define the colors for the gradient
    colors = ["#ff0000", "#ffffff", "#a9def9"]  # Red, White, light Blue

    # Create a custom colormap
    custom_cmap_red_white_blue = mcolors.LinearSegmentedColormap.from_list("CustomRedWhiteBlue", colors)
    
    # Define the color map as a diverging gradient from red to blue with white at zero
    cmap = custom_cmap_red_white_blue   

    # Use TwoSlopeNorm to set the midpoint to zero or other value
    # The good combination for TD value: min=-30, center=0, max=30
    # The good combination for sensitivity value: min=0, center=30, max=60
    # The good combination for PD value: min=-35, center=0, max=2
    norm = mcolors.TwoSlopeNorm(vmin=min, vcenter=center, vmax=max)
    
    for x in range(dat.shape[0]):
        #print('Archetype '+str(x+1))
        
        # Generate random data with both positive and negative values
        #data = dat[x].reshape((9,9))
        data = dat[x].reshape((td,td))

        # Create grey-only colormap
        grey_cmap = ListedColormap(["grey"])
    
        # Create masks
        mask_regular = np.full_like(data, False, dtype=bool)
        mask_grey = np.full_like(data, True, dtype=bool)
    
        # Mark blind spot coordinates (zero-indexed!)
        # x=7, y=3 → row=3, col=7; x=7, y=4 → row=4, col=7
        blind_coords = [(3, 7), (4, 7)]
        for r, c in blind_coords:
            mask_regular[r, c] = True   # Exclude from main heatmap
            mask_grey[r, c] = False     # Include only these in grey heatmap
        
        # Create the heatmap with the custom colormap and normalization
        plt.figure(figsize=(fwidth, fheight))
        sns.heatmap(data, annot=annotasi, fmt=".2f", cmap=cmap, norm=norm, linewidths=0.5, cbar=False, mask=mask_regular)

        if bsgrey == True:
            # Overlay grey for blind spot
            sns.heatmap(data, annot=annotasi, fmt=".2f", cmap=grey_cmap, norm=norm, linewidths=0.5, cbar=False, mask=mask_grey)

        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        
        # Saving the figure.
        plt.savefig(fname+'_'+str(x+1)+".png")
        
        # Display the plot
        plt.show()


# Function to visualize the result in r rows, c columns. 
def plot_result_onebyone_LR(fname, dat, dat2, min=-30, center=0, max=30, td=9, fwidth=4, fheight=3, annotasi=False):
    # td = 9 for display in 9x9 for 24-2. For 30-2 and 10-2, it will be td=10 (10x10)
    # fwidth and fheight for 24-2 and 30-2 is 15 and 10. But for 10-2 it is 12 and 7
    # Define the colors for the gradient
    colors = ["#ff0000", "#ffffff", "#a9def9"]  # Red, White, light Blue

    # Create a custom colormap
    custom_cmap_red_white_blue = mcolors.LinearSegmentedColormap.from_list("CustomRedWhiteBlue", colors)
    
    # Define the color map as a diverging gradient from red to blue with white at zero
    cmap = custom_cmap_red_white_blue

    # Use TwoSlopeNorm to set the midpoint to zero or other value
    # The good combination for TD value: min=-30, center=0, max=30
    # The good combination for sensitivity value: min=0, center=30, max=60
    # The good combination for PD value: ymin=-35, center=0, max=2
    norm = mcolors.TwoSlopeNorm(vmin=min, vcenter=center, vmax=max)
    
    for x in range(dat.shape[0]):
        #print('Archetype '+str(x+1))

        # Set up the figure with two subplots (one for each eye)
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))

        dataleft = dat[x].reshape((td,td))
        dataright = dat2[x].reshape((td,td))
        
        # Left Eye heatmap
        sns.heatmap(dataleft, ax=axes[0], annot=annotasi, fmt=".2f", cmap=cmap, norm=norm, linewidths=0.5, cbar=False)
        axes[0].set_title(f'Left Eye - Row {x}')
        #axes[0].set_xlabel('Columns')
        #axes[0].set_ylabel('Rows')
        axes[0].set_xticks([])  # Remove x-axis ticks
        axes[0].set_yticks([])  # Remove y-axis ticks
    
        # Right Eye heatmap
        sns.heatmap(dataright, ax=axes[1], annot=annotasi, fmt=".2f", cmap=cmap, norm=norm, linewidths=0.5, cbar=False)
        axes[1].set_title(f'Right Eye - Row {x}')
        #axes[1].set_xlabel('Columns')
        #axes[1].set_ylabel('Rows')
        axes[1].set_xticks([])  # Remove x-axis ticks
        axes[1].set_yticks([])  # Remove y-axis ticks
        
        # Create the heatmap with the custom colormap and normalization
        #plt.figure(figsize=(fwidth, fheight))
        #sns.heatmap(data, annot=annotasi, fmt=".2f", cmap=cmap, norm=norm, linewidths=0.5, cbar=False)
        
        # Saving the figure.
        plt.savefig(fname+'_'+str(x+1)+".png")
        
        # Display the plot
        plt.show()


from sklearn.metrics.pairwise import cosine_similarity

# Function to measure cosine similarity between archetypes in 2 files. Must have the same row. Then visualize in heatmap.
def cos_similarity_hmap(fname1, fname2, name1, name2, ffigure):
    # Load the CSV files
    #file1_path = "Test3/oriAA_UWHVF_TD_16x.csv"  # Update with actual path
    #file2_path = "Test3/pypchaAA_UWHVF_TD_16x.csv"  # Update with actual path
    file1_path = fname1  # Update with actual path
    file2_path = fname2  # Update with actual path
    
    data1 = pd.read_csv(file1_path)
    data2 = pd.read_csv(file2_path)
    
    # Ensure both datasets have the same number of rows and columns
    common_rows = min(data1.shape[0], data2.shape[0])
    common_columns = min(data1.shape[1], data2.shape[1])
    
    data1_aligned = data1.iloc[:common_rows, :common_columns]
    data2_aligned = data2.iloc[:common_rows, :common_columns]
    
    # Compute row-wise cosine similarity
    row_wise_cosine_sim = cosine_similarity(data1_aligned, data2_aligned)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        row_wise_cosine_sim, 
        annot=True, 
        cmap="coolwarm", 
        fmt=".2f",
        xticklabels=np.arange(1, common_rows + 1),  # Set x-axis labels from 1
        yticklabels=np.arange(1, common_rows + 1)   # Set y-axis labels from 1
    )

    plt.xlabel(name2)
    plt.ylabel(name1)
    plt.title("Cosine Similarity Heatmap Between "+name1+ " and " +name2)

    # Saving the figure.
    plt.savefig(ffigure+'.png')
    
    # Show the heatmap
    plt.show()

# Function to find archetype max value for each data point and find the distribution of data point based on archetypes
def max_and_distribution_archetype(finput1, foutput1, foutput2):
    #finput 1 is the alfa file
    #fouput1 will be the archetype max value for each data point file
    #fouput2 will be the distribution of data point based on archetype
    
    # Load the CSV file
    #file_path = "Test3/oriAA_UWHVF_TD_alfa_16.csv"  # Update with your actual file path
    file_path = finput1
    df = pd.read_csv(file_path)
    
    # Compute the maximum value for each column and its row index
    max_values = df.max(numeric_only=True)  # Get max values for numeric columns
    max_indices = df.idxmax(numeric_only=True)  # Get indices of max values
    
    # Create a DataFrame with results
    max_df = pd.DataFrame({"Max Value": max_values, "Archetype": max_indices+1})  #max_indices plus 1 so that it matches the archetype 1-20
    
    # Save the results to a CSV file
    max_df.to_csv(foutput1, index=False)
    
    print(f"Processed file saved as {foutput1}")
    
    # Load the CSV file
    file_path = foutput1  # Update with your actual file path
    df = pd.read_csv(file_path)
    
    # Ensure the "Archetype" column exists
    if "Archetype" not in df.columns:
        raise ValueError("The file does not contain an 'Archetype' column.")
    
    # Count occurrences of each Archetype
    archetype_counts = df["Archetype"].value_counts()
    
    # Calculate percentages
    archetype_percentages = (archetype_counts / archetype_counts.sum()) * 100
    
    # Create a DataFrame with results
    archetype_summary = pd.DataFrame({
        "Count": archetype_counts,
        "Percentage": archetype_percentages
    })
    
    # Save the results to a CSV file
    output_file = foutput2
    archetype_summary.to_csv(output_file, index=True)
    
    print(f"Processed file saved as {foutput2}.")


def plot_duration_rss(methods, base_dir, filename_template, output_path, plot_title, vis="Duration", ylab="Seconds", show=True, j=1):
    """
    Plot duration or RSS curves with NaN-padded start for multiple methods.

    Parameters:
    - methods: list of method names (used in both folder and filename)
    - base_dir: base path to the folder containing the method subfolders
    - filename_template: template for the filename with a placeholder for method name
    - output_path: where to save the plot
    - plot_title: title of the plot
    - vis: "Duration" or "RSS"
    - ylab: "Seconds" for Duration, "RSS" for RSS comparison
    - show: plot is shown (True) or not (False)
    - j: folder enumerate
    """

    durations = []
    labels = []
    
    for method in methods:
        file_path = os.path.join(base_dir, f"{method}{j}", filename_template.format(method))
        df = pd.read_excel(file_path)
        padded = [np.nan] + list(df[vis])
        durations.append(padded)
        labels.append(method)

    # Create x-axis values
    x_values = np.arange(1, len(durations[0]) + 1)

    # Plotting
    plt.figure(figsize=(7, 5))
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    markers = ['o', 'x'] * len(methods)  # alternate markers

    for i, duration in enumerate(durations):
        plt.plot(x_values, duration, label=labels[i], marker=markers[i], color=colors[i % len(colors)])

    plt.xticks(np.arange(1, len(durations[0]) + 1))
    plt.xlabel("Number of archetypes")
    plt.ylabel(ylab)
    plt.title(plot_title)
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    if show:
        plt.show()

    print(f"✅ Plot saved to {output_path}")

def plot_simplex_file(k=2, mytitle="My scatter plot", size = 10, color_dot="black", xalfa=[], ffigure = "plot_alfa", plot_args = {}, grid_on = True):
    
    """
    # group_color = None, color = None, marker = None, size = None
    group_color:    
        
        Dimension:      n_data x 1
        
        Description:    Contains the category of data point.
    """
    alfa = np.array(xalfa)

    labels = ('A'+str(i + 1) for i in range(k))
    rotate_labels=True
    label_offset=0.10
    data = alfa.T
    scaling = False
    sides=k
    
    basis = np.array(
                [
                    [
                        np.cos(2*_*pi/sides + 90*pi/180),
                        np.sin(2*_*pi/sides + 90*pi/180)
                    ] 
                    for _ in range(sides)
                ]
            )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T,basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data,basis)

    
    
    fig = plt.figure(figsize=(size,size))
    ax = fig.add_subplot(111)

    for i,l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i,0]
        y = basis[i,1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = (angle + 180) % 360 # mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
                x*(1 + label_offset),
                y*(1 + label_offset),
                l,
                horizontalalignment='center',
                verticalalignment='center',
                rotation=angle
            )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)
    
    
    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False                        
#                
            if not (ignore):                    
                lst_ax_0.append(basis[i,0] + [0,])
                lst_ax_1.append(basis[i,1] + [0,])
                lst_ax_0.append(basis[j,0] + [0,])
                lst_ax_1.append(basis[j,1] + [0,])


    
    ax.plot(lst_ax_0,lst_ax_1, color='#FFFFFF',linewidth=1, alpha = 0.5, zorder=1)
    
    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    for _ in range(sides):
        lst_ax_0.append(basis[_,0] + [0,])
        lst_ax_1.append(basis[_,1] + [0,])

    lst_ax_0.append(basis[0,0] + [0,])
    lst_ax_1.append(basis[0,1] + [0,])

#    
    ax.plot(lst_ax_0,lst_ax_1,linewidth=1, zorder=2) #, **edge_args ) 
    
    if len(plot_args) == 0:
        ax.scatter(newdata[:,0], newdata[:,1], color=color_dot, zorder=3, alpha=0.5)
    else:
        if ('marker' in plot_args):   
            marker_vals = plot_args['marker'].values
            marker_unq = np.unique(marker_vals)                
            
            for marker in marker_unq:
                row_idx = np.where(marker_vals == marker)
                tmp_arg = {}
                for keys in plot_args:
                    if (keys!= 'marker'):
                        tmp_arg[keys] = plot_args[keys].values[row_idx]
                
                ax.scatter(newdata[row_idx,0],newdata[row_idx,1], **tmp_arg, marker =  marker, alpha=0.5, zorder=3)
        else:
            ax.scatter(newdata[:,0], newdata[:,1], **plot_args, marker = 's', zorder=3, alpha=0.5)

    # Add title
    plt.title(mytitle)
    
    # Saving the figure.
    plt.savefig(ffigure+'.png')

    plt.show()
