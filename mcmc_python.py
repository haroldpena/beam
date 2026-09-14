import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt
from astropy.io import fits

# --- Configuration ---
INPUT_FITS = "group1/tmp/m20251129_00078_01_backoff.fits"
N_WALKERS = 32
N_BURN = 10  # 100
N_PROD = 20  # 500

LABELS = ["fwhm1", "fwhm2", "amp2", "x_cen", "y_cen", "peak", "back"]
NDIM = len(LABELS)

# --- Load Data ---
def load_data(fits_path):
    """Load FITS data and return image array."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        # Handle potential extra dimensions
        if data.ndim > 2:
            data = np.squeeze(data)
        return data

# --- Model Function ---
def double_gaussian_2d(shape, fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back, radius):
    """
    Create 2D double Gaussian model.
    
    Parameters:
    - fwhm1, fwhm2: FWHM in arcsec for two components
    - amp2: Amplitude ratio of second to first Gaussian
    - x_cen, y_cen: Center offset in arcsec
    - peak: Peak amplitude of first Gaussian
    - back: Background level
    - radius: Jupiter radius in arcsec (for scaling)
    """
    ny, nx = shape
    
    # Create coordinate grids (centered at image center)
    y, x = np.mgrid[0:ny, 0:nx]
    y = y - ny / 2.0 + y_cen
    x = x - nx / 2.0 + x_cen
    
    # Convert FWHM to sigma (FWHM = 2.355 * sigma)
    sigma1 = fwhm1 / 2.355
    sigma2 = fwhm2 / 2.355
    
    # Calculate radial distance
    r_sq = x**2 + y**2
    
    # Two Gaussian components
    gauss1 = peak * np.exp(-r_sq / (2 * sigma1**2))
    gauss2 = peak * amp2 * np.exp(-r_sq / (2 * sigma2**2))
    
    model = gauss1 + gauss2 + back
    return model

# --- Likelihood Function ---
def ln_prob(params, data, radius):
    """Log probability function for MCMC."""
    fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back = params
    
    # Priors
    if not (0 < fwhm1 < 100 and 0 < fwhm2 < 200 and 0 < amp2 < 1.0 and 
            -10 < x_cen < 10 and -10 < y_cen < 10 and peak > 0 and back >= 0):
        return -np.inf
    
    # Generate model
    model = double_gaussian_2d(data.shape, fwhm1, fwhm2, amp2, x_cen, y_cen, peak, back, radius)
    
    # Calculate chi-squared (assuming uniform noise for now)
    # Use robust sigma estimation from data
    residuals = data - model
    sigma = np.nanstd(residuals)
    
    if sigma == 0 or not np.isfinite(sigma):
        return -np.inf
    
    chi2 = np.nansum(residuals**2) / sigma**2
    
    # Return log likelihood
    return -0.5 * chi2

# --- Main Execution ---
if __name__ == "__main__":
    # Load data
    print("Loading data...")
    data = load_data(INPUT_FITS)
    print(f"Data shape: {data.shape}, range: [{np.nanmin(data):.3e}, {np.nanmax(data):.3e}]")
    
    # Jupiter radius (constant for now)
    jup_radius = 18.6283
    
    # Initial guess
    initial_guess = [6.0, 30.0, 0.05, 0.01, 0.01, 0.1, 0.01]
    
    # Initialize walkers
    pos = initial_guess + 1e-4 * np.random.randn(N_WALKERS, NDIM)
    
    print(f"Starting MCMC for {INPUT_FITS} using Radius={jup_radius}...")
    
    # Create sampler
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM, ln_prob, args=[data, jup_radius])
    
    # Burn-in
    print("Running burn-in...")
    pos, prob, state = sampler.run_mcmc(pos, N_BURN, progress=True)
    sampler.reset()
    
    # Production
    print("Running production...")
    sampler.run_mcmc(pos, N_PROD, progress=True)
    
    # Analyze results
    samples = sampler.get_chain(flat=True)
    best_fit = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    print("\n--- Final MCMC Best Fit ---")
    with open("mcmc_results.txt", "w") as f:
        for i, label in enumerate(LABELS):
            res_str = f"{label}: {best_fit[i]:.4f} +/- {stds[i]:.4f}"
            print(res_str)
            f.write(res_str + "\n")
    
    # Corner plot
    fig = corner.corner(samples, labels=LABELS, truths=best_fit)
    fig.savefig("mcmc_dynamic_radius.png")
    print("\nCorner plot saved to 'mcmc_dynamic_radius.png'")
