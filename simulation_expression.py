# -*- coding: utf-8 -*-
"""
Created on Sat May 24 02:08:17 2025

@author: Tejaswini
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math
import os

# --- Parameters for User Adjustment ---
# IMPORTANT: EXPERIMENT WITH THESE TWO SCALING FACTORS!
# 1. Scaling factor for production rates derived from data.
# If your input average expression values are high (e.g., 100s-1000s),
# try very SMALL values here (e.g., 0.001, 0.0001, or even less)
# to reduce basal production and allow regulatory effects to be seen.
PRODUCTION_RATE_SCALING_FACTOR = 100 # MODIFIED - Start with a very small value

# 2. Global scaling factor for ALL interaction strengths defined in 'edges'.
# Increase this to make regulatory effects stronger if they are still too weak.
# Try values like 10, 50, 100, 500, or more.
STRENGTH_SCALING_FACTOR = 100.0  # MODIFIED - Start with a significantly larger value

# Global scaling factor for input data (e.g., raw expression values from CSV)
# This is useful if your CSV input expression values are very large or small
# and you want to rescale them without affecting production rate scaling separately
DATA_INPUT_SCALING_FACTOR = 0.0001
# Degradation rate for all genes.
DEGRADATION_RATE = 0.1
# Minimum absolute percentage change in AUC to display.
MIN_ABS_PERC_CHANGE_TO_DISPLAY = 0.01
# --- End Parameters for User Adjustment ---

# 1. Define GRN Edges 
edges = [ 
    ("SREBF1", "FASN", 1, 5.0, 3.0, 0.7),
    ("SREBF1", "ACACA", 1, 5.0, 3.0, 0.7),
    ("SREBF1", "HMGCS1", 1, 5.0, 3, 0.7),
    ("SREBF1", "ACLY", 1, 5.0, 3.0, 0.7),
    ("SREBF1", "GPAM", 1, 5.0, 3.0, 0.7),
    ("SREBF1", "LRP1", -1, 2.5, 3.0, 0.7),
    ("SREBF2", "SCD", 1, 5.0, 2.5, 0.7),
    ("SREBF2", "ACLY", 1, 5.0, 3.0, 0.7),
    ("SREBF2", "SREBF1", 1, 2.0, 2.0, 1.0),
    ("MLXIPL", "DGAT2", 1, 1.2, 2.0, 1.0),
    ("MLXIPL", "DGAT1", 1, 1.2, 2.0, 1.0),
    ("MLXIPL", "FASN", 1, 1.2, 2.0, 1.0),
    ("MLXIPL", "NR1H3", 1, 1.2, 2.0, 1.0),
    ("MLXIPL", "SCD", 1, 1.2, 2.0, 1.0),
    ("MLXIPL", "THRSP", 1, 1.2, 2.0, 1.0),
    ("MLXIPL", "ELOVL", 1, 1.2, 2.0, 1.0),
    ("NR1H3", "SREBF1", 1, 2.0, 2.0, 1.0),
    ("NR1H2", "SREBF1", 1, 2.0, 2.0, 1.0),
    ("PPARGC1B", "NR1H2", 1, 1.2, 1.5, 1.0),
    ("INSIG1", "SCAP", -1, 3.0, 2.0, 1.0),      
    ("INSIG2", "SCAP", -1, 1.8, 2.0, 1.0),      
    ("INSIG1", "SREBF1", -1, 1.2, 2.5, 1.0),    
    ("INSIG1", "SREBF2", -1, 1.2, 2.5, 1.0),    
    ("SREBF1", "INSIG1", 1, 3.0, 2.0, 1.0), 
    ("SREBF2", "INSIG1", 1, 3.0, 2.0, 1.0), 
    ("SCAP", "SREBF1", 1, 3.0, 2.0, 1.0),       
    ("SCAP", "SREBF2", 1, 3.0, 2.0, 1.0),       
    ("SREBF1", "SREBF1", 1, 2.0, 2.5, 1.0),     
    ("SREBF2", "SREBF2", 1, 2.0, 2.5, 1.0),      
    ("SREBF2", "LDLR", 1, 3.0, 2.5, 0.7),
    ("SREBF2", "HMGCR", 1, 3.0, 2.5, 0.7),
    ("SREBF1", "ELOVL", 1, 2.8, 2.0, 0.7),
    ("SREBF1", "DGAT1", 1, 2.5, 2.0, 0.7),
    ("SREBF1", "DGAT2", 1, 2.5, 2.0, 0.7),
    ("PPARGC1B", "NR1H3", 1, 1.5, 1.5, 1.0),
    ("FASN", "SREBF1", -1, 1.5, 1.5, 1.0),         
    ("HMGCR", "SREBF2", -1, 2.0, 2.0, 1.0),        
    ("HMGCR", "SCAP", -1, 2.0, 1.8, 1.0),          
    ("SCD", "SREBF1", -1, 1.5, 1.5, 1.0),          
    ("ELOVL", "SREBF1", -1, 1.2, 1.5, 1.0),       
    ("ACACA", "SREBF1", -1, 1.2, 1.5, 1.0),        
    ("DGAT2", "SREBF1", -1, 1.2, 1.5, 1.0),        
    ("HMGCS1", "SCAP", -1, 1.8, 2.0, 1.0),         
]         

genes = sorted(list(set(g for edge_tuple in edges for g in edge_tuple[:2])))
gene_index = {g: i for i, g in enumerate(genes)}
n_genes = len(genes)

# 2. Hill Functions
def hill_act(x, K, n):
    x_safe = max(0.0, x) 
    return (x_safe ** n) / (K ** n + x_safe ** n + 1e-9)

def hill_rep(x, K, n):
    x_safe = max(0.0, x) 
    return (K ** n) / (K ** n + x_safe ** n + 1e-9)

# 3. ODE System
def grn_dynamics(y, t, current_edges, current_gene_index, current_knockout_genes, current_prod_rates, degr=DEGRADATION_RATE, strength_scale=STRENGTH_SCALING_FACTOR):
    dy = np.zeros_like(y)
    
    for target_gene_name in genes:
        i = current_gene_index[target_gene_name]
        
        if target_gene_name in current_knockout_genes:
            dy[i] = -degr * max(0.0, y[i]) 
            continue

        basal_production_rate = current_prod_rates.get(target_gene_name, 0.0)
        accumulated_regulation = 0.0
        
        for src, tgt, effect, strength, n_coeff, k_factor in current_edges:
            if tgt != target_gene_name:
                continue
            
            effective_strength = strength * strength_scale 

            if src in current_knockout_genes: 
                continue 
            
            x_level = max(0.0, y[current_gene_index[src]])
            
            if abs(effective_strength) < 1e-9: # Effectively no interaction if strength is zero
                continue

            # IMPORTANT: If effective_strength becomes very large due to scaling,
            # calculated_K can become very small. This makes the Hill function highly sensitive (switch-like).
            # This might be desirable or might need k_factor to be adjusted too.
            calculated_K = abs(k_factor / effective_strength) 
            calculated_K = max(1e-9, calculated_K) 
            
            regulatory_term_value = 0.0
            if effect == 1: 
                regulatory_term_value = effective_strength * hill_act(x_level, calculated_K, n_coeff)
            else: # effect == -1
                regulatory_term_value = -effective_strength * hill_rep(x_level, calculated_K, n_coeff)
            
            accumulated_regulation += regulatory_term_value
        
        total_production_rate = basal_production_rate + accumulated_regulation
        effective_production = max(0.0, total_production_rate)
        degradation_term_val = degr * max(0.0, y[i])
        dy[i] = effective_production - degradation_term_val
            
    return dy

# 4. Load Production Rates
def load_production_rates(csv_file_path, gene_names_from_edges, 
                          data_input_scale=1.0, # Added to accept the new global factor
                          prod_rate_specific_scale=1.0): 
    try:
        df = pd.read_csv(csv_file_path, index_col=0) 
        df.index = df.index.map(str)
        
        # Apply overall data input scaling first
        df_scaled = df * data_input_scale  # <--- NEW: Scale all data from CSV
        
        avg_expression_for_rates = df_scaled.mean(axis=1) 

        loaded_rates = {}
        for g_name in gene_names_from_edges:
            # Use scaled average for rate calculation, then apply specific production rate scaling
            # Also scale the default value
            val = avg_expression_for_rates.get(g_name, 0.01 * data_input_scale) 
            loaded_rates[g_name] = val * prod_rate_specific_scale # This uses PRODUCTION_RATE_SCALING_FACTOR
        return loaded_rates
    except FileNotFoundError:
        print(f"ERROR: File not found at {csv_file_path}")
        return None
    except Exception as e:
        print(f"ERROR loading production rates: {e}")
        return None

# 5. Run Simulation
def simulate(csv_path_param, knockout_genes_param=[], selected_genes_to_plot=None, 
             current_data_input_scale=1.0): # Added current_data_input_scale parameter
    t = np.linspace(0, 100, 500)
    
    # Pass DATA_INPUT_SCALING_FACTOR (as current_data_input_scale) to load_production_rates
    prod_rates_base = load_production_rates(csv_path_param, genes, 
                                            data_input_scale=current_data_input_scale, 
                                            prod_rate_specific_scale=PRODUCTION_RATE_SCALING_FACTOR)
    if prod_rates_base is None:
        print("Halting simulation: Error in loading production rates.")
        return

    try:
        expression_df_orig = pd.read_csv(csv_path_param, index_col=0) 
        expression_df_orig.index = expression_df_orig.index.map(str)
        
        # Apply overall data input scaling for initial conditions as well
        expression_df_scaled = expression_df_orig * current_data_input_scale # <--- NEW: Scale data for y0
        
        avg_expression_scaled = expression_df_scaled.mean(axis=1)
        # Scale the default value for .get() too
        default_initial_val = 0.01 * current_data_input_scale 
        y0_initial = np.array([avg_expression_scaled.get(g, default_initial_val) for g in genes])

    except Exception as e:
        print(f"ERROR reading for initial conditions (using scaled 0.01 for all): {e}")
        y0_initial = np.full(n_genes, 0.01 * current_data_input_scale) # Scale fallback too

    # --- Your DEBUG print statements should now reflect these scalings ---
    print("\n---- DEBUG: Key Parameters & Values ----")
    print(f"DATA_INPUT_SCALING_FACTOR (for this run): {current_data_input_scale}") # Show the run-specific scale
    print(f"PRODUCTION_RATE_SCALING_FACTOR (applied after data input scale): {PRODUCTION_RATE_SCALING_FACTOR}")
    print(f"STRENGTH_SCALING_FACTOR (global): {STRENGTH_SCALING_FACTOR}")
    print(f"DEGRADATION_RATE (global): {DEGRADATION_RATE}")
    
    genes_to_debug = ["SREBF1", "FASN", "SCD", "ACACA", "ACLY", "SREBF2", "INSIG1"] 
    print("\n-- Initial Values (y0) & Final Basal Production Rates (after ALL scaling) --")
    for gene_name_dbg in genes_to_debug:
        if gene_name_dbg in gene_index:
            idx_dbg = gene_index[gene_name_dbg]
            # prod_rates_base already has both scaling factors applied in load_production_rates
            rate_dbg = prod_rates_base.get(gene_name_dbg, "NOT_FOUND") 
            print(f"Gene: {gene_name_dbg:<10} | y0: {y0_initial[idx_dbg]:<10.4e} | Final Basal Prod Rate: {rate_dbg if isinstance(rate_dbg, str) else f'{rate_dbg:<10.4e}'}")
            
            for src, tgt, effect, strength, _, _ in edges:
                if src == "SREBF1" and tgt == gene_name_dbg: # Example for SREBF1
                    scaled_strength = strength * STRENGTH_SCALING_FACTOR
                    print(f"    SREBF1->{tgt} | Edge Strength: {strength:.2f}, Final Scaled Strength: {scaled_strength:.2f}, Effect: {effect}")
                elif src == "SREBF2" and tgt == gene_name_dbg: # Example for SREBF2
                    scaled_strength = strength * STRENGTH_SCALING_FACTOR
                    print(f"    SREBF2->{tgt} | Edge Strength: {strength:.2f}, Final Scaled Strength: {scaled_strength:.2f}, Effect: {effect}")
        else:
            print(f"Gene: {gene_name_dbg:<10} | Not in model's 'genes' list (check 'edges').")
    print("----------------------------------------------------------\n")

    # --- Rest of your simulate function (ODE solving, plotting calls) remains the same ---
    norm_args = (edges, gene_index, [], prod_rates_base, DEGRADATION_RATE, STRENGTH_SCALING_FACTOR)
    sol_norm = odeint(grn_dynamics, y0_initial.copy(), t, args=norm_args)

    y0_ko = y0_initial.copy()
    for g_ko in knockout_genes_param:
        if g_ko in gene_index:
            y0_ko[gene_index[g_ko]] = 0.0 

    ko_args = (edges, gene_index, knockout_genes_param, prod_rates_base, DEGRADATION_RATE, STRENGTH_SCALING_FACTOR)
    sol_ko = odeint(grn_dynamics, y0_ko, t, args=ko_args)

    df_norm = pd.DataFrame(sol_norm, columns=genes, index=t)
    df_ko = pd.DataFrame(sol_ko, columns=genes, index=t)

    plot_results(df_norm, df_ko, t, knockout_genes_param, selected_genes=selected_genes_to_plot)
    plot_auc_comparison_and_save_excel(df_norm, df_ko, t, genes, knockout_genes_param, csv_path_param)

# 6. Plotting Time Course Results 
def plot_results(df1, df2, t_pts, ko_list, selected_genes=None):
    genes_to_plot = selected_genes if selected_genes is not None else df1.columns
    rows = math.ceil(len(genes_to_plot) / 2)
    fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 3.2), sharex=True, squeeze=False)
    axes = axes.ravel()

    # Compute global y-axis limits
    global_min = float('inf')
    global_max = float('-inf')
    for g in genes_to_plot:
        all_vals = np.concatenate([df1[g].values, df2[g].values])
        global_min = min(global_min, np.min(all_vals))
        global_max = max(global_max, np.max(all_vals))
    global_min = min(0, global_min)  # Ensure it starts at 0 if values are all positive

    for i, g_name in enumerate(genes_to_plot):
        ax = axes[i]
        normal_expr = df1[g_name]
        ko_expr = df2[g_name]

        # Plot Normal and KO curves
        ax.plot(t_pts, normal_expr, label="Normal", color="blue")
        ax.plot(t_pts, ko_expr, label=f"KO {', '.join(ko_list)}", color="red", linestyle="--")

        # Shade area between curves
        ax.fill_between(t_pts, normal_expr, ko_expr, where=(normal_expr > ko_expr),
                        interpolate=True, color="blue", alpha=0.2, label="↑ Normal")
        ax.fill_between(t_pts, normal_expr, ko_expr, where=(ko_expr > normal_expr),
                        interpolate=True, color="red", alpha=0.2, label="↑ KO")

        ax.set_title(g_name)
        ax.set_ylabel("Expression")
        ax.set_ylim(global_min, global_max)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
        
    fig.suptitle("Thymic Epithelium\nGene Expression Dynamics, KO of {', '.join(ko_list)}", fontsize=14)
    # Set x-labels only on last row
    start_idx_last_row = (rows - 1) * 2
    for ax_idx in range(start_idx_last_row, start_idx_last_row + 2):
        if ax_idx < len(genes_to_plot) and axes[ax_idx].has_data(): 
            axes[ax_idx].set_xlabel("Time")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

# 7. Calculate AUC, Plot Comparison, and Save to Excel (Unchanged from last version)
def plot_auc_comparison_and_save_excel(df_normal, df_knockout, time_points, gene_list, ko_gene_names, input_csv_path):
    auc_normal_values = []
    auc_ko_values = []
    signed_diff_values = []     
    signed_perc_diff_values = []

    for gene in gene_list:
        auc_n = np.trapz(df_normal[gene], x=time_points)
        auc_k = np.trapz(df_knockout[gene], x=time_points)
        
        auc_normal_values.append(auc_n)
        auc_ko_values.append(auc_k)
        
        signed_diff = auc_k - auc_n
        signed_diff_values.append(signed_diff)
        
        current_signed_perc_diff = np.nan 

        if abs(auc_n) < 1e-9: 
            if abs(auc_k) < 1e-9:
                current_signed_perc_diff = 0.0
            else: 
                current_signed_perc_diff = np.sign(auc_k) * 200.0 
        else:
            raw_signed_perc_diff = (signed_diff / auc_n) * 100
            if abs(raw_signed_perc_diff) < MIN_ABS_PERC_CHANGE_TO_DISPLAY:
                current_signed_perc_diff = 0.0
            else:
                current_signed_perc_diff = raw_signed_perc_diff
        
        signed_perc_diff_values.append(current_signed_perc_diff)

    ko_auc_col_name = f'KO {", ".join(ko_gene_names)} AUC' if ko_gene_names else 'AUC Condition 2'
    diff_col_name = f'Difference ({ko_auc_col_name} - Normal AUC)'
    perc_change_col_name = 'Percentage Change (%)'
    
    auc_data_for_df = {
        'Gene': gene_list,
        'Normal AUC': auc_normal_values,
        ko_auc_col_name: auc_ko_values,
        diff_col_name: signed_diff_values,
        perc_change_col_name: signed_perc_diff_values
    }
    df_auc_results = pd.DataFrame(auc_data_for_df)
    # Filter df_auc_results to only include genes that are in the model's 'genes' list.
    df_auc_results = df_auc_results[df_auc_results['Gene'].isin(genes)]


    ordered_columns = ['Gene', 'Normal AUC', ko_auc_col_name, 
                       diff_col_name, perc_change_col_name]
    df_auc_results = df_auc_results[ordered_columns]

    ko_filename_suffix = "_".join(ko_gene_names).replace(" ", "_") if ko_gene_names else "NoKO"
    excel_short_filename = f"auc_comparison_KO_{ko_filename_suffix}.xlsx"
    
    try:
        input_file_directory = os.path.dirname(input_csv_path)
        if not input_file_directory : 
            input_file_directory = "." 
        output_excel_full_path = os.path.join(input_file_directory, excel_short_filename)
        df_auc_results.to_excel(output_excel_full_path, index=False)
        print(f"\nAUC data successfully saved to: {output_excel_full_path}")
    except Exception as e:
        print(f"\nError saving AUC data to Excel: {e}")

    plot_df = df_auc_results[(abs(df_auc_results['Normal AUC']) > 1e-6) | \
                             (abs(df_auc_results[ko_auc_col_name]) > 1e-6) | \
                             (abs(df_auc_results[perc_change_col_name]) > MIN_ABS_PERC_CHANGE_TO_DISPLAY )
                            ].copy()
    plot_df.sort_values(by=perc_change_col_name, key=abs, ascending=False, inplace=True)
    
    genes_for_plot = plot_df['Gene'].tolist()
    auc_normal_plot = plot_df['Normal AUC'].tolist()
    auc_ko_plot = plot_df[ko_auc_col_name].tolist()

    if not genes_for_plot:
        print("No significant AUC data to plot based on current filters.")
        return

    x = np.arange(len(genes_for_plot))  
    width = 0.35  

    fig, ax = plt.subplots(figsize=(max(12, 0.4 * len(genes_for_plot)), 7))
    rects1 = ax.bar(x - width/2, auc_normal_plot, width, label='Normal AUC', color='deepskyblue')
    rects2 = ax.bar(x + width/2, auc_ko_plot, width, label=ko_auc_col_name, color='salmon')

    ax.set_ylabel('Area Under Curve (AUC)')
    ax.set_title(f'AUC Comparison: Normal vs KO {", ".join(ko_gene_names)}' if ko_gene_names else 'AUC Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(genes_for_plot, rotation=75, ha="right", fontsize=8)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.tight_layout() 
    plt.show()

# 8. Main
if __name__ == '__main__':
    # --- CONFIGURE YOUR RUN HERE ---
    
    # Path to your input CSV file. 
    # Ensure 'gene_name' is the FIRST column.
    # CHANGE THIS PATH TO YOUR "NORMAL" OR "TUMOR" CSV AS NEEDED
    csv_data_path = r"C:\Users\Tejaswini\Desktop\cancerrrrrr\mtp_2\analysis\geodataset\b cell lymphoma\normal\GSE281832_count_matrix.csv"

    # Set DATA_INPUT_SCALING_FACTOR at the top of the script (around line 10)
    # For a "normal" sample run, you might have DATA_INPUT_SCALING_FACTOR = 1.0
    # For a "tumor" sample run, you might set DATA_INPUT_SCALING_FACTOR = 0.1 (or 0.05, 0.02 etc.)
    #   based on how much higher tumor values are compared to normal.
    # The value set globally at the top will be used here.
    
    print(f"### Running for dataset: {os.path.basename(csv_data_path)} ###")
    print(f"### Using DATA_INPUT_SCALING_FACTOR for this run: {DATA_INPUT_SCALING_FACTOR} ###") # Uses the global value
    
    if not os.path.exists(csv_data_path):
        print(f"ERROR: Input CSV file not found at: {csv_data_path}")
    else:
        ko_target_genes = ["SREBF1"] 
        selected_to_plot = ["SREBF1", "FASN", "SCD", "ACACA", "ACLY", "INSIG1", "SREBF2", "HMGCR", "MLXIPL"]
        
        print(f"Starting simulation with KO of: {ko_target_genes}")
        print(f"Global PRODUCTION_RATE_SCALING_FACTOR = {PRODUCTION_RATE_SCALING_FACTOR}")
        print(f"Global STRENGTH_SCALING_FACTOR = {STRENGTH_SCALING_FACTOR}")
        print(f"Global DEGRADATION_RATE = {DEGRADATION_RATE}")
        print("--- Please check the DEBUG output below for initial values and rates ---")
        
        # Pass the global DATA_INPUT_SCALING_FACTOR to the simulate function
        simulate(csv_data_path, 
                 knockout_genes_param=ko_target_genes, 
                 selected_genes_to_plot=selected_to_plot,
                 current_data_input_scale=DATA_INPUT_SCALING_FACTOR) 