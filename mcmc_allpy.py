import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import fftconvolve
from astropy.io import fits
from astropy.wcs import WCS
from multiprocessing import Pool, cpu_count
import sys
from glob import glob
import pandas as pd

# --- Model Functions (from user's gau2fit translation) ---
def create_planet_model(shape, radius_px, center_px):
    """Creates a sharp-edged circular disc (planet model)."""
    ny, nx = shape
    y, x = np.ogrid[:ny, :nx]
    dist_sq = (x - center_px[0])**2 + (y - center_px[1])**2
    planet = np.where(dist_sq <= radius_px**2, 1.0, 0.0)
    # Normalize to unit sum as per Starlink description
    return planet / np.sum(planet) if np.sum(planet) > 0 else planet

def gaussian_2d_kernel(shape, fwhm_px, amplitude=1.0):
    """Generates a 2D Gaussian kernel."""
    ny, nx = shape
    sigma = fwhm_px / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    y, x = np.ogrid[:ny, :nx]
    # Center the kernel for convolution
    cy, cx = ny // 2, nx // 2
    g = amplitude * np.exp(-((x - cx)**2 + (y - cy)**2) / (2.0 * sigma**2))
    return g

def calculate_model(data_shape, fwhm1, fwhm2, amp2, x_c, y_c, peak, back, 
                    radius_px, pixsize):
    """
    Calculate the model: (planet ⊗ beam) * peak + background.
    Matches C implementation with amp1=1.0 fixed.
    """
    # Build Beam: First Gaussian has amplitude 1.0 (FIXED)
    # Second Gaussian has amplitude amp2 (relative to first)
    beam = gaussian_2d_kernel(data_shape, fwhm1 / pixsize, amplitude=1.0)
    beam += gaussian_2d_kernel(data_shape, fwhm2 / pixsize, amplitude=amp2)
    
    # Normalize beam to unit sum (as per C code)
    beam_sum = np.sum(beam)
    if beam_sum > 0:
        beam /= beam_sum
    
    # Create Planet disc (normalized)
    planet = create_planet_model(data_shape, radius_px, (x_c, y_c))
    
    # Convolve planet with beam, then scale by peak and add background
    model = fftconvolve(planet, beam, mode='same') * peak + back
    
    return model

# --- Likelihood Function ---


def ln_prob(params, data, radius_arcsec, pixsize, fixed_sigma):
    """Log probability function for MCMC with penalties to avoid degeneracies."""
    fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back = params
    
    # Tighter priors based on expected physical values
    if not (2.0 < fwhm1 < 15.0):  # Narrower range for first component
        return -np.inf
    if not (15.0 < fwhm2 < 60.0):  # Second component should be broader
        return -np.inf
    if not (0.0 < amp2 < 0.1):  # amp2 should be small relative to amp1=1.0
        return -np.inf
    if not (0 < x_cen < data.shape[1] and 0 < y_cen < data.shape[0]):
        return -np.inf
    if not (10.0 < peak < 1.0e5):  # Much higher range - peak gets diluted by convolution
        return -np.inf
    if not (-5.0 < back < 5.0):  # Background should be near zero
        return -np.inf
    
    # Physical constraint: Second Gaussian MUST be broader than first
    if fwhm2 <= fwhm1:
        return -np.inf
    
    # Calculate model
    radius_px = radius_arcsec / pixsize
    model = calculate_model(data.shape, fwhm1, fwhm2, amp2, x_cen, y_cen, 
                           peak, back, radius_px, pixsize)
    
    # Calculate residuals
    residuals = data - model
    
    # Use fixed noise estimate (can be improved with real noise map)
    # For now, use MAD (Median Absolute Deviation) for robustness
    # mad = np.nanmedian(np.abs(residuals - np.nanmedian(residuals)))
    # sigma = 1.4826 * mad  # Convert MAD to std
    
    if fixed_sigma == 0 or not np.isfinite(fixed_sigma):
        return -np.inf
    
    # Chi-squared
    chi2 = np.nansum(residuals**2) / fixed_sigma**2
    
    # Add penalties to avoid degeneracies (as per C code comments)
    penalty = 0.0
    
    # Penalty 1: Discourage similar FWHMs (prevents degeneracy)
    fwhm_ratio = fwhm1 / fwhm2
    if fwhm_ratio > 0.5:  # Getting too similar
        penalty += 10.0 / (1.0 - fwhm_ratio)**2
    
    # Penalty 2: Discourage very wide second components
    array_size = np.sqrt(data.shape[0] * data.shape[1]) * pixsize
    if fwhm2 > 0.5 * array_size:
        penalty += 10.0 * (fwhm2 / array_size)**2
    
    # Penalty 3: Discourage large amplitude second components
    if amp2 > 0.2:
        penalty +=  100.0 * amp2**2
    
    # Log likelihood with penalties
    return -0.5 * (chi2 + penalty)

# --- Diagnostic Plotting Functions ---
def plot_diagnostics(data, best_params, stds, radius_arcsec, pixsize, output_prefix="fit_diagnostics", planet_name="Jupiter", rest_freq=None):
    """
    Create comprehensive diagnostic plots for the MCMC fit.
    
    Parameters:
    - data: 2D array of observed data
    - best_params: [fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back]
    - stds: errors for [fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back]
    - radius_arcsec: Planet radius in arcsec
    - pixsize: Pixel size in arcsec/pixel
    - output_prefix: Prefix for output files
    """
    fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back = best_params
    fwhm1_err, fwhm2_err, amp2_err, x_cen_err, y_cen_err, peak_err, back_err = stds
    radius_px = radius_arcsec / pixsize
    
    # Calculate full model
    full_model = calculate_model(data.shape, fwhm1, fwhm2, amp2, x_cen, y_cen, 
                                 peak, back, radius_px, pixsize)
    
    # Calculate individual CONVOLVED Gaussian components for visualization
    # --- CORRECTED PLOTTING SECTION ---
    planet = create_planet_model(data.shape, radius_px, (x_cen, y_cen))
    
    # 1. Calculate the "Volume" (flux contribution) of each component
    # Volume of 2D Gaussian approx proportional to Amplitude * FWHM^2
    vol1 = 1.0 * fwhm1**2
    vol2 = amp2 * fwhm2**2
    total_vol = vol1 + vol2
    
    # 2. Calculate the Weight (fraction of total light) for each
    weight1 = vol1 / total_vol
    weight2 = vol2 / total_vol
    
    # 3. Component 1: Normalized shape * Total Flux * WEIGHT 1
    beam1 = gaussian_2d_kernel(data.shape, fwhm1 / pixsize, amplitude=1.0)
    beam1 /= np.sum(beam1) # Shape normalized to 1
    gauss1_convolved = fftconvolve(planet, beam1, mode='same') * peak * weight1
    
    # 4. Component 2: Normalized shape * Total Flux * WEIGHT 2
    beam2 = gaussian_2d_kernel(data.shape, fwhm2 / pixsize, amplitude=1.0) # Note: amp=1 here for shape
    beam2 /= np.sum(beam2) # Shape normalized to 1
    gauss2_convolved = fftconvolve(planet, beam2, mode='same') * peak * weight2
    
    # Now, gauss1_convolved + gauss2_convolved will equal full_model
    
    residuals = data - full_model
    
    # Convert center to arcsec (from center of image)
    x_cen_arcsec = (x_cen - data.shape[1]/2.0) * pixsize
    y_cen_arcsec = (y_cen - data.shape[0]/2.0) * pixsize
    
    # Create figure with all subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f"Model Fit: {fits_file}", fontsize=16)
    
    # ===== PLOT 1: 2D Contours (Data + Gaussians) =====
    ax1 = plt.subplot(2, 3, 1)
    extent = [0, data.shape[1]*pixsize, 0, data.shape[0]*pixsize]
    
    # Plot data contours
    levels_data = np.linspace(np.nanpercentile(data, 80), np.nanpercentile(data, 99), 10)
    cs_data = ax1.contour(data, levels=levels_data, extent=extent, colors='black', linewidths=1.5, alpha=0.7)
    ax1.clabel(cs_data, inline=True, fontsize=8)
    
    # Overlay convolved Gaussian 1 (narrow) in blue
    levels_g1 = np.linspace(np.nanpercentile(gauss1_convolved, 90), np.nanpercentile(gauss1_convolved, 99), 5)
    ax1.contour(gauss1_convolved, levels=levels_g1, extent=extent, colors='blue', linestyles='--', linewidths=2, alpha=0.8)
    
    # Overlay convolved Gaussian 2 (broad) in orange  
    levels_g2 = np.linspace(np.nanpercentile(gauss2_convolved, 70), np.nanpercentile(gauss2_convolved, 99), 5)
    ax1.contour(gauss2_convolved, levels=levels_g2, extent=extent, colors='orange', linestyles=':', linewidths=2, alpha=1)
    
    ax1.set_xlabel('X (arcsec)')
    ax1.set_ylabel('Y (arcsec)')
    ax1.set_title('Data (black) + Gauss1 (blue) + Gauss2 (orange)')
    ax1.grid(True, alpha=0.3)
    
    # ===== PLOT 2: 2D Residuals =====
    ax2 = plt.subplot(2, 3, 2)
    vmax = np.nanpercentile(np.abs(residuals), 95)
    im = ax2.imshow(residuals, origin='lower', extent=extent, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax2, label='Residual')
    ax2.set_xlabel('X (arcsec)')
    ax2.set_ylabel('Y (arcsec)')
    ax2.set_title(f'Residuals (RMS={np.nanstd(residuals):.2f})')
    
    # ===== PLOT 3: 1D X-projection =====
    ax3 = plt.subplot(2, 3, 3)
    x_axis = np.arange(data.shape[1]) * pixsize
    
    # Collapse along Y axis
    data_x = np.nansum(data, axis=0)
    model_x = np.nansum(full_model, axis=0)
    gauss1_x = np.nansum(gauss1_convolved, axis=0)
    gauss2_x = np.nansum(gauss2_convolved, axis=0)
    resid_x = np.sqrt(np.nanmean(residuals**2, axis=0))
    
    ax3.plot(x_axis, data_x, 'k-', linewidth=2, label='Data')
    ax3.plot(x_axis, model_x, 'g--', linewidth=1.5, label='Model')
    ax3.plot(x_axis, gauss1_x, 'b:', linewidth=2.0, alpha=1, label=f'G1 (FWHM={fwhm1:.1f}")')
    ax3.plot(x_axis, gauss2_x, 'orange', linewidth=2.0, alpha=1, label=f'G2 (FWHM={fwhm2:.1f}")')
    ax3.set_xlabel('X (arcsec)')
    ax3.set_ylabel('Integrated Intensity')
    ax3.set_title('X-axis Projection')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Residuals below
    ax3b = ax3.twinx()
    ax3b.bar(x_axis, resid_x, width=pixsize*0.8, alpha=0.3, color='gray', label='Residual')
    ax3b.set_ylabel('Residual', color='gray')
    ax3b.tick_params(axis='y', labelcolor='gray')
    
    # ===== PLOT 4: 1D Y-projection =====
    ax4 = plt.subplot(2, 3, 4)
    y_axis = np.arange(data.shape[0]) * pixsize
    
    # Collapse along X axis
    data_y = np.nansum(data, axis=1)
    model_y = np.nansum(full_model, axis=1)
    gauss1_y = np.nansum(gauss1_convolved, axis=1)
    gauss2_y = np.nansum(gauss2_convolved, axis=1)
    resid_y = np.sqrt(np.nanmean(residuals**2, axis=1))
    
    ax4.plot(y_axis, data_y, 'k-', linewidth=2, label='Data')
    ax4.plot(y_axis, model_y, 'g--', linewidth=1.5, label='Model')
    ax4.plot(y_axis, gauss1_y, 'b:', linewidth=2.0, alpha=1, label=f'G1')
    ax4.plot(y_axis, gauss2_y, 'orange', linewidth=2.0, alpha=1, label=f'G2')
    ax4.set_xlabel('Y (arcsec)')
    ax4.set_ylabel('Integrated Intensity')
    ax4.set_title('Y-axis Projection')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Residuals below
    ax4b = ax4.twinx()
    ax4b.bar(y_axis, resid_y, width=pixsize*0.8, alpha=0.3, color='gray', label='Residual')
    ax4b.set_ylabel('Residual', color='gray')
    ax4b.tick_params(axis='y', labelcolor='gray')
    
    # ===== PLOT 5: 3D Surface Plot =====
    ax5 = plt.subplot(2, 3, 5, projection='3d')
    X, Y = np.meshgrid(x_axis, y_axis)
    
    # Plot data surface
    surf1 = ax5.plot_surface(X, Y, data, cmap='winter', alpha=0.7, edgecolor='none')
    
    # Plot model wireframe
    ax5.plot_wireframe(X, Y, full_model, color='orange', linewidth=1, alpha=1)
    ax5.grid(True, alpha=0.5)
    
    ax5.set_xlabel('X (arcsec)')
    ax5.set_ylabel('Y (arcsec)')
    ax5.set_zlabel('Intensity')
    ax5.set_title('3D: Data (surface) + Model (wireframe)')
    
    # ===== PLOT 6: Additional info text =====
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = f"""
MCMC Fit Results:
━━━━━━━━━━━━━━━━
FWHM Primary Beam = {fwhm1:.2f} +/- {fwhm1_err:.4f} arcsec
FWHM Error Beam = {fwhm2:.2f} +/- {fwhm2_err:.4f} arcsec
Amplitude Primary Beam = 1.00
Amplitude Error Beam = {amp2:.3f} +/- {amp2_err:.4f}

Center Position:
━━━━━━━━━━━━━━━━
x = {x_cen:.2f} +/- {x_cen_err:.4f} pix = {x_cen_arcsec:.2f}" offset
y = {y_cen:.2f} +/- {y_cen_err:.4f} pix = {y_cen_arcsec:.2f}" offset

Intensity:
━━━━━━━━━━━━━━━━
peak = {peak:.2f} +/- {peak_err:.4f}
background = {back:.2f} +/- {back_err:.4f}

Fit Quality:
━━━━━━━━━━━━━━━━
RMS residual = {np.nanstd(residuals):.3f}
Reduced χ² ≈ {np.nansum(residuals**2)/residuals.size:.2f}

Input:
━━━━━━━━━━━━━━━━
Planet name = {planet_name}
Planet radius = {radius_arcsec:.2f} arcsec
Pixel size = {pixsize:.2f} arcsec/pix
Rest Frequency = {f"{rest_freq:.4f} GHz" if rest_freq is not None else "N/A"}
    """
    
    ax6.text(0.1, 0.5, info_text, fontsize=11, verticalalignment='center', 
             family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(plot_dir+"/"+f"{output_prefix}.png", dpi=150, bbox_inches='tight')
    print(f"\nDiagnostic plots saved to: {plot_dir}/{output_prefix}.png")
    plt.close()



def run_mcmc(INPUT_NDF, planet_name, planet_radius, rest_freq=None):
    # --- Configuration ---
    N_WALKERS = 200
    N_BURN = 500
    N_PROD = 1000

    LABELS = ["fwhm1", "fwhm2", "amp2", "x_cen", "y_cen", "peak", "back"]
    NDIM = len(LABELS)
    # Load Data (FITS file converted from NDF)
    print("Loading data...")
    fits_file = INPUT_NDF + ".fits"
    hdul = fits.open(fits_file)
    data = hdul[0].data
    header = hdul[0].header

    
    # Handle extra dimensions
    if data.ndim > 2:
        data = np.squeeze(data)
    
    print(f"Data shape: {data.shape}, range: [{np.nanmin(data):.3e}, {np.nanmax(data):.3e}]")
    
    # Get pixel scale (arcsec/pixel)
    try:
        wcs = WCS(header)
        pixsize = np.abs(header.get('CDELT1', 1.0)) * 3600.0
    except:
        pixsize = 2.0  # default fallback (arcsec/pixel)
    
    print(f"Pixel size: {pixsize:.2f} arcsec/pixel")
    
    # Initial guess (closer to expected values from C routine)
    # Expected: fwhm1~9, fwhm2~33, amp2~0.067
    # Peak needs to be high because convolution dilutes it - start with 10x observed peak
    init_x = data.shape[1] / 2.0
    init_y = data.shape[0] / 2.0
    total_flux_estimate = np.nansum(data)
    initial_guess = [9.0, 33.0, 0.07, init_x, init_y, total_flux_estimate, 0.0]
    
    # Initialize walkers with small perturbations
    pos = initial_guess + 1e-2 * np.random.randn(N_WALKERS, NDIM) * np.array([1, 5, 0.01, 1, 1, 0.01*initial_guess[5], 0.01])
    
    print(f"Starting MCMC for {INPUT_NDF} using Radius={planet_radius} arcsec...")
    print(f"Initial guess: {initial_guess}")
    
    # Estimate noise from a corner of the image known to be empty
    bg_region = data[0:30, 0:30] # Adjust indices to pick an empty corner
    global_sigma = 1.4826 * np.nanmedian(np.abs(bg_region - np.nanmedian(bg_region)))
    print(f"Fixed Noise Estimate (sigma): {global_sigma}")
    
    # Setup multiprocessing
    n_cpus = cpu_count()
    print(f"Using {n_cpus} CPU cores for parallel execution")
    
    # Create sampler with multiprocessing pool
    with Pool(n_cpus) as pool:
        sampler = emcee.EnsembleSampler(N_WALKERS, NDIM, ln_prob, 
                                       args=[data, planet_radius, pixsize, global_sigma],
                                       pool=pool)
    
        # Burn-in
        print("\nRunning burn-in...")
        pos, prob, state = sampler.run_mcmc(pos, N_BURN, progress=True)
        print(f"Acceptance fraction during burn-in: {np.mean(sampler.acceptance_fraction):.3f}")
        sampler.reset()
        
        # Production
        print("\nRunning production...")
        sampler.run_mcmc(pos, N_PROD, progress=True)
        print(f"Acceptance fraction during production: {np.mean(sampler.acceptance_fraction):.3f}")
    
    # Analyze results (outside the pool context)
    samples = sampler.get_chain(flat=True)
    best_fit = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    print("\n--- Final MCMC Best Fit ---")
    with open(f"{data_path}/mcmc_results.txt", "w") as f:
        for i, label in enumerate(LABELS):
            res_str = f"{label}: {best_fit[i]:.4f} +/- {stds[i]:.4f}"
            print(res_str)
            f.write(res_str + "\n")
    
    # Corner plot
    import os
    base_name = os.path.basename(INPUT_NDF)
    fig = corner.corner(samples, labels=LABELS, truths=best_fit)
    fig.suptitle(f"Corner plot: {fits_file}", fontsize=16)
    fig.savefig(f"{data_path}/plots/mcmc_dynamic_radius_"+base_name+".png")
    print("\nCorner plot saved to 'mcmc_dynamic_radius_"+base_name+".png'")
    
    # Generate diagnostic plots
    print("\nGenerating diagnostic plots...")
    plot_diagnostics(data, best_fit, stds, planet_radius, pixsize, output_prefix="mcmc_fit_diagnostics_"+base_name, planet_name="Jupiter", rest_freq=rest_freq)
    return best_fit, stds



import subprocess
import os

def convert_ndf_to_fits(sdf_path):
    """
    Converts an NDF (.sdf) file to FITS (.fits) using Starlink's ndf2fits.
    Uses subprocess to call tcsh and source the necessary environment.
    """
    fits_path = sdf_path.replace(".sdf", ".fits")
    if os.path.exists(fits_path):
        print(f"FITS file already exists: {fits_path}")
        return fits_path
    
    print(f"Converting {sdf_path} to FITS...")
    
    # TCsh script to source starlink and run conversion
    # Note: we use 'convert' and then 'ndf2fits'
    tcsh_cmd = f"""
    source /Users/haroldpena/starlink-2025A/star-2025A/etc/login
    source /Users/haroldpena/starlink-2025A/star-2025A/etc/cshrc
    convert
    ndf2fits in={sdf_path} out={fits_path}
    """
    
    try:
        result = subprocess.run(['tcsh'], input=tcsh_cmd.encode(), capture_output=True, check=True)
        if os.path.exists(fits_path):
            print(f"Successfully converted to {fits_path}")
            return fits_path
        else:
            print(f"Conversion failed for {sdf_path}. Output: {result.stdout.decode()} {result.stderr.decode()}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion of {sdf_path}: {e}")
        print(f"Stdout: {e.stdout.decode() if e.stdout else ''}")
        print(f"Stderr: {e.stderr.decode() if e.stderr else ''}")
        return None


def get_ndf_header_values(sdf_path):
    """
    Extracts specific FITS keywords from an NDF (.sdf) file using Starlink's fitslist.
    Returns a dictionary of the requested keywords.
    """
    keywords = ["OBJECT", "IFFREQ", "LOFREQS", "OBS_SB","SB_MODE","ALIGN_DX", "ALIGN_DY", "DATE-OBS"]
    
    # TCsh script to source starlink and run fitslist (part of KAPPA)
    tcsh_cmd = f"""
    source /Users/haroldpena/starlink-2025A/star-2025A/etc/login
    source /Users/haroldpena/starlink-2025A/star-2025A/etc/cshrc
    kappa
    fitslist in={sdf_path}
    """
    
    try:
        result = subprocess.run(['tcsh'], input=tcsh_cmd.encode(), capture_output=True, check=True)
        output = result.stdout.decode()
        header_data = {}
        
        for line in output.splitlines():
            if "=" in line:
                parts = line.split("=")
                key = parts[0].strip()
                if key in keywords:
                    # Extract value, removing comments (after /) and quotes
                    value = parts[1].split("/")[0].strip().strip("'").strip()
                    header_data[key] = value
        
        return header_data
    except subprocess.CalledProcessError as e:
        print(f"Error running fitslist on {sdf_path}: {e}")
        return None

def calculate_restfrequency(IFFREQ, LOFREQS, OBS_SB, SB_MODE):
    """
    Calculates the rest frequency for JCMT heterodyne instrumentation.
    Supports 2SB (Sideband Separating) and SSB (Single Sideband) modes.
    """
    invalid = []
    if IFFREQ == 0: invalid.append("IFFREQ")
    if LOFREQS == 0: invalid.append("LOFREQS")
    if OBS_SB == '': invalid.append("OBS_SB")
    if SB_MODE == '': invalid.append("SB_MODE")
    
    if invalid:
        print(f"The following parameters are 0 or empty: {', '.join(invalid)}")
        return None

    if SB_MODE.upper() not in ['2SB', 'SSB']:
        raise ValueError(f"SB_MODE '{SB_MODE}' is not recognized. Expected '2SB' or 'SSB'.")

    if OBS_SB.upper() == 'USB':
        return LOFREQS + IFFREQ
    elif OBS_SB.upper() == 'LSB':
        if SB_MODE.upper() == '2SB':
            return LOFREQS - IFFREQ
        elif SB_MODE.upper() == 'SSB':
            return LOFREQS + IFFREQ
    else:
        raise ValueError("OBS_SB must be 'USB' or 'LSB'")

def get_planet_semidiameter(date_str, time_str, planet="jupiter"):
    """
    Uses Starlink's FLUXES package to get the semi-diameter of a planet.
    date_str: 'DD MM YY'
    time_str: 'HH MM SS'
    """
    outfile = "fluxes.dat"
    if os.path.exists(outfile):
        os.remove(outfile)

    tcsh_cmd = f"""
    source /Users/haroldpena/starlink-2025A/star-2025A/etc/login
    source /Users/haroldpena/starlink-2025A/star-2025A/etc/cshrc
    fluxes
    fluxes pos=yes flu=yes screen=no ofl=yes now=no date={date_str} time={time_str} planet={planet} outfile={outfile} apass=no quiet=yes
    """

    try:
        subprocess.run(['tcsh'], input=tcsh_cmd.encode(), capture_output=True, check=True)
        if os.path.exists(outfile):
            print(f'{outfile} was created')
            semi_diameter = None
            with open(outfile, 'r') as f:
                for line in f:
                    if "Semi-diameter" in line:
                        # Expected format: "Semi-diameter = 21.36 arcsecs"
                        parts = line.split("=")
                        if len(parts) > 1:
                            semi_diameter = float(parts[1].strip().split()[0])
                            break
            os.remove(outfile)
            return semi_diameter
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        print(f"Error retrieving semi-diameter for {planet}: {e}")
    
    return None


# --- Main Execution ---

if __name__ == "__main__":
    # Find all .sdf files
    data_path = "/Users/haroldpena/ownCloud/Work/working/kuntur/beam/20260207"#20260112" #20260112 20260118
    tcsh_output_path = f"{data_path}/group1/tmp"
    sdf_files = glob(f"{tcsh_output_path}/*backoff.sdf")
    plot_dir = f"{data_path}/plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    print(f"Found {len(sdf_files)} NDF files.")
    
    # Initialize results dataframe
    LABELS = ["fwhm1", "fwhm2", "amp2", "x_cen", "y_cen", "peak", "back"]
    # Header keywords to extract
    HEADER_KEYS = ["OBJECT", "IFFREQ", "LOFREQS", "OBS_SB", "SB_MODE", "ALIGN_DX", "ALIGN_DY", "DATE-OBS"]
    columns = ["file"] + LABELS + [f"{l}_err" for l in LABELS] + HEADER_KEYS + ["RESTFREQ"]
    bf_df = pd.DataFrame(columns=columns)
    
    for sdf_file in sdf_files:
        print(f"\n" + "="*50)
        print(f"Processing: {sdf_file}")
        keywords = get_ndf_header_values(sdf_file)
        print('keywords: ', keywords)
        #'OBJECT': 'JUPITER', 'DATE-OBS': '2025-11-30T10:40:01'     date_str: 'DD MM YY'    time_str: 'HH MM SS'
        date_part, time_part = keywords['DATE-OBS'].split('T')
        y, m, d = date_part.split('-')
        date_str = f"{d} {m} {y[2:]}"
        time_str = time_part.replace(':', ' ')
        date_str = f"\\'{date_str}\\'"
        time_str = f"\\'{time_str}\\'"
        
        planet_name = keywords['OBJECT'].lower()
        
        planet_radius = get_planet_semidiameter(date_str, time_str, planet=planet_name)
        print('date_str: ', date_str)
        print('time_str: ', time_str)
        print('planet_name: ', planet_name)
        print('planet_radius: ', planet_radius)

        rest_freq = None
        try:
            iffreq = float(keywords.get('IFFREQ', 0))
            lofreq = float(keywords.get('LOFREQS', 0))
            obs_sb = keywords.get('OBS_SB', '')
            sb_mode = keywords.get('SB_MODE', '')
            rest_freq = calculate_restfrequency(iffreq, lofreq, obs_sb, sb_mode)
            print(f"Calculated RESTFREQ: {rest_freq} GHz")
        except Exception as e:
            print(f"Could not calculate RESTFREQ: {e}")
        
        # 1. Convert NDF to FITS
        fits_file = convert_ndf_to_fits(sdf_file)
        if not fits_file:
            print(f"Skipping {sdf_file} due to conversion failure.")
            continue
            
        # 2. Run MCMC
        # Input to run_mcmc should be the prefix (without .fits)
        input_prefix = fits_file.replace(".fits", "")
        
        try:
            best_fit, stds = run_mcmc(input_prefix, planet_name, planet_radius, rest_freq=rest_freq)
            
            # Store results
            header_values = [keywords.get(k, None) for k in HEADER_KEYS]
            row = [os.path.basename(input_prefix)] + list(best_fit) + list(stds) + header_values + [rest_freq]
            bf_df = pd.concat([bf_df, pd.DataFrame([row], columns=columns)], ignore_index=True) 
            
        except Exception as e:
            print(f"Error running MCMC for {sdf_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    print("\n" + "="*50)
    print("Summary of MCMC results:")
    if not bf_df.empty:
        print(bf_df.to_string(index=False))
        bf_df.to_csv(f"{data_path}/mcmc_results_batch.csv", index=False)
        print(f"\nResults saved to {data_path}/mcmc_results_batch.csv")
    else:
        print("No successful fits were completed.")



