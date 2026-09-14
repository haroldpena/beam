import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.optimize import curve_fit

# Configure Plotting Style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

# def assign_frequency(filename):
#     """
#     Assign frequency based on file naming pattern.
#     Files ending in 01-04_backoff -> 691.5 GHz
#     Files ending in 05-08_backoff -> 702.5 GHz
    
#     NOTE: In future versions, this should come directly from the CSV 
#     (frequency column should be added during MCMC processing).
#     """
#     # Extract the number before "_backoff"
#     match = re.search(r'_(\d{2})_backoff', filename)
#     if match:
#         num = int(match.group(1))
#         if 1 <= num <= 4:
#             return 691.5
#         elif 5 <= num <= 8:
#             return 702.5
#     return None



def inverse_variance_weighted_mean(values, errors):
    """
    Calculate inverse-variance weighted mean and its error.
    
    weighted_mean = sum(value_i / variance_i) / sum(1 / variance_i)
    weighted_error = sqrt(1 / sum(1 / variance_i))
    
    Parameters:
    - values: array of measured values
    - errors: array of measurement errors
    
    Returns:
    - weighted_mean, weighted_error
    """
    # Convert to numpy arrays
    values = np.array(values)
    errors = np.array(errors)
    
    # Calculate variances
    variances = errors**2
    
    # Inverse variance weights
    weights = 1.0 / variances
    
    # Weighted mean
    weighted_mean = np.sum(values * weights) / np.sum(weights)
    
    # Error on weighted mean
    weighted_error = np.sqrt(1.0 / np.sum(weights))
    
    return weighted_mean, weighted_error

def fwhm_model(freq, a, b):
    """Model function: FWHM = a + (b / frequency)"""
    return a + (b / freq)

def fit_and_plot_results(results_df, data_path):
    """
    Fit the FWHM data to the model and plot the results.
    """
    print("\n" + "="*60)
    print("Performing Fits and Generating Plots...")
    print("="*60)
    
    freqs = results_df['RESTFREQ'].values
    
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax2 = None
    try:
        trans_file = '/Users/haroldpena/ownCloud/Work/working/kuntur/beam/mcmc/mko_transmission_kuntur.dat'
        # Read file assuming whitespace delimiter
        # Col 0: Freq, Col 2: Trans (at 0.5mm vapor?)
        td = pd.read_csv(trans_file, delim_whitespace=True, header=None)
        
        ax2 = ax.twinx()
        ax2.fill_between(td[0], 0, td[2]*100, color='lightgrey', alpha=0.3, label='Atm. Trans.')
        # Or just plot line? User said "Plot ... in light grey". Line implies plot.
        ax2.plot(td[0], td[2]*100, color='lightgrey', linewidth=1, zorder=0)
        
        ax2.set_ylabel('Transmission (%)', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        
        # Ensure main plot is on top
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        
    except Exception as e:
        print(f"Warning: Could not plot transmission data: {e}")
    
    # Colors and markers
    colors = {'fwhm1': 'blue', 'fwhm2': 'red'}
    labels = {'fwhm1': 'Primary Beam', 'fwhm2': 'Error Beam'}
    markers = {'fwhm1': 'o', 'fwhm2': 's'}
    
    # Iterate over FWHM parameters
    for param in ['fwhm1', 'fwhm2']:
        y_data = results_df[f'{param}_weighted'].values
        y_err = results_df[f'{param}_weighted_err'].values
        
        # Perform Fit
        # p0: initial guess. a ~ 0, b ~ fwhm * freq 
        try:
            popt, pcov = curve_fit(fwhm_model, freqs, y_data, sigma=y_err, absolute_sigma=True, p0=[0, y_data.mean()*freqs.mean()])
            perr = np.sqrt(np.diag(pcov))
            
            a_fit, b_fit = popt
            a_err, b_err = perr
            
            print(f"\n{param.upper()} Fit Results:")
            print(f"  a = {a_fit:.4e} +/- {a_err:.4e}")
            print(f"  b = {b_fit:.4e} +/- {b_err:.4e}")
            
            # Generate model line for plotting
            x_plot = np.linspace(609, 715, 100)
            y_plot = fwhm_model(x_plot, *popt)
            
            # Plot Fit
            fit_label = fr"{labels[param]} (arcsec): ${a_fit:.2f} + \dfrac{{{b_fit/690.0:.2f} \times 690}}{{\text{{Frequency (GHz)}}}}$"
            
            ax.plot(x_plot, y_plot, linestyle='--', color=colors[param], label=fit_label)
            
        except Exception as e:
            print(f"Fit failed for {param}: {e}")

        # Plot Data
        ax.errorbar(freqs, y_data, yerr=y_err, fmt=markers[param], color=colors[param], capsize=5, label=f"{labels[param]} Data")

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('FWHM (arcsec)')
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plot_file = f"{data_path}/fwhm_vs_freq_fit.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {plot_file}")
    plt.close()

def main():
    # Read the CSV files
    root_path = "/Users/haroldpena/ownCloud/Work/working/kuntur/beam"
    datafolders = ['20260112', '20260118', '20260207']
    data_path = root_path

    print(f"Reading mcmc_results_batch.csv from {len(datafolders)} folders...")
    df = pd.concat([pd.read_csv(f"{root_path}/{folder}/mcmc_results_batch.csv") for folder in datafolders], ignore_index=True)
    
    # Add frequency column
    print("Assigning frequencies based on file naming...")
    df['RESTFREQ'] = round(df['RESTFREQ'], 1)
    
    # Display the dataframe with frequencies
    print("\nData with assigned frequencies:")
    print(df[['file', 'fwhm1', 'fwhm1_err', 'fwhm2', 'fwhm2_err', 'amp2', 'amp2_err', 'RESTFREQ']].to_string(index=False))
    
    # Parameters to calculate weighted means for
    params = ['fwhm1', 'fwhm2', 'amp2']
    
    # Group by RESTFREQ and calculate inverse-variance weighted means
    results = []
    
    for freq, group in df.groupby('RESTFREQ'):
        print(f"\n{'='*60}")
        print(f"RESTFREQ: {freq} GHz ({len(group)} observations)")
        print(f"{'='*60}")
        
        result_row = {'RESTFREQ': freq, 'n_obs': len(group)}
        
        for param in params:
            values = group[param].values
            errors = group[f'{param}_err'].values
            
            # Calculate weighted mean
            w_mean, w_error = inverse_variance_weighted_mean(values, errors)
            
            result_row[f'{param}_weighted'] = w_mean
            result_row[f'{param}_weighted_err'] = w_error
            
            # print(f"\n{param.upper()}:")
            # print(f"  Individual values: {values}")
            # print(f"  Individual errors: {errors}")
            def format_value(value, error):
                return f"{value:.3g} +/- {error:.3g}"
            
            print(f"  Weighted mean: {format_value(w_mean, w_error)}")
        
        results.append(result_row)
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    column_order = ['RESTFREQ', 'n_obs']
    for param in params:
        column_order.extend([f'{param}_weighted', f'{param}_weighted_err'])
    results_df = results_df[column_order]
    
    # Save results
    output_file = f"{root_path}/mcmc_weighted_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n{'='*60}")
    print(f"Weighted results saved to: {output_file}")
    print(f"{'='*60}")
    print("\nFinal weighted results:")

    display_df = results_df[['RESTFREQ', 'n_obs']].copy()
    for param in params:
        display_df[param] = results_df.apply(
            lambda r: f"{r[f'{param}_weighted']:.3g} +/- {r[f'{param}_weighted_err']:.3g}", axis=1
        )
    print(display_df.to_string(index=False))
    
    # Perform Fit and Plot
    fit_and_plot_results(results_df, data_path)
    
    return results_df

if __name__ == "__main__":
    main()
