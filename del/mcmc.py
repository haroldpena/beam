import os
import subprocess
import numpy as np
import emcee
import corner
import matplotlib.pyplot as plt

# --- Configuration ---
INPUT_NDF = "group1/tmp/m20251129_00078_01_backoff.sdf"
N_WALKERS = 32
N_BURN = 10#100
N_PROD = 20#500

LABELS = ["fwhm1", "fwhm2", "amp2", "x_cen", "y_cen", "peak", "back"]
NDIM = len(LABELS)

def get_jupiter_radius(ndf_path):
    # Use absolute paths to binaries to bypass shell alias issues
    bin_dir = "/Users/haroldpena/starlink-2025A/star-2025A/bin"
    star_etc = "/Users/haroldpena/starlink-2025A/star-2025A/etc"
    
    # Updated command string: bypasses the 'kappa' alias and calls the binary directly
    cmd_base = f"source {star_etc}/cshrc; source {star_etc}/login; "
    
    try:
        # Use KAPPA's fitsval directly from the bin directory
        ut_date = subprocess.check_output(f"{cmd_base} {bin_dir}/kappa/fitsval {ndf_path} UTDATE", shell=True, executable="/bin/tcsh").decode().strip()
        ut_start = subprocess.check_output(f"{cmd_base} {bin_dir}/kappa/fitsval {ndf_path} UTSTART", shell=True, executable="/bin/tcsh").decode().strip()
        
        y, m, d = ut_date.split('-')
        
        # Call fluxes
        fluxes_cmd = (
            f"{cmd_base} {bin_dir}/smurf/fluxes pos=no flu=yes screen=no ofl=no now=no "
            f"date='{d} {m} {y}' time='{ut_start}' planet=jupiter outfile=fluxes.dat apass=no quiet=yes; "
            f"{bin_dir}/kappa/parget semi_diam fluxes"
        )
        
        radius = float(subprocess.check_output(fluxes_cmd, shell=True, executable="/bin/tcsh").decode().strip())
        return radius
    except Exception as e:
        print(f"Starlink Error: {e}")
        return 18.6283

def run_gau2fit(params, radius):
    f1, f2, a2, xc, yc, pk, bk = params
    pid = os.getpid()
    logfile = f"tmp_log_{pid}.txt"
    model_ndf = f"tmp_mod_{pid}"
    
    # Use absolute path to the binary directly
    # This avoids the "Bad : modifier" caused by sourcing cshrc inside the loop
    star_bin = "/Users/haroldpena/starlink-2025A/star-2025A/bin/smurf/gau2fit"
    
    # We pass the environment variables directly to the subprocess
    my_env = os.environ.copy()
    my_env["ADAM_USER"] = os.path.expanduser(f"~/.adam_{pid}") # Prevents file locking
    
    # Construct command without 'source' inside the string
    # We use 'initcentre' as a string to avoid shell expansion issues
    cmd = [
        star_bin,
        f"in={INPUT_NDF}",
        f"out={model_ndf}",
        "fittwo=yes",
        f"fwhm1={f1}",
        f"fwhm2={f2}",
        f"amp2={a2}",
        f"peak={pk}",
        f"back={bk}",
        f"initcentre='{xc},{yc}'",
        f"radius={radius}",
        f"logfile={logfile}",
        "quiet=yes"
    ]

    try:
        # Run as a list (more secure/stable than a string)
        subprocess.run(cmd, env=my_env, check=True, 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        rms = 1e9 
        if os.path.exists(logfile):
            with open(logfile, 'r') as f:
                for line in f:
                    if "RMS" in line:
                        # Extract last element (the RMS value)
                        rms = float(line.split()[-1])
            os.remove(logfile)
        
        # Cleanup temporary files
        for ext in [".sdf", ".pnt", ".ref"]:
            tmp_file = f"{model_ndf}{ext}"
            if os.path.exists(tmp_file): os.remove(tmp_file)
            
        return rms
    except:
        return 1e9


# --- Likelihood Wrapper ---
def ln_prob(params, radius):
    # Prior check
    f1, f2, a2, xc, yc, pk, bk = params
    if not (0 < f1 < 100 and 0 < f2 < 200 and 0 < a2 < 1.0 and -10 < xc < 10 and -10 < yc < 10):
        return -np.inf
    
    rms = run_gau2fit(params, radius)
    if rms >= 1e9: return -np.inf
    return -0.5 * (rms**2)

# --- Execution ---
if __name__ == "__main__":
    # 1. Get Jupiter Radius
    jup_radius = get_jupiter_radius(INPUT_NDF)

    # 2. Setup MCMC
    initial_guess = [6.0, 30.0, 0.05, 0.01, 0.01, 0.1, 0.01]
    pos = initial_guess + 1e-4 * np.random.randn(N_WALKERS, NDIM)

    print(f"Starting MCMC for {INPUT_NDF} using Radius={jup_radius}...")
    
    # Pass the jup_radius as an extra argument (args=[jup_radius])
    sampler = emcee.EnsembleSampler(N_WALKERS, NDIM, ln_prob, args=[jup_radius])

    print("Running burn-in...")
    pos, prob, state = sampler.run_mcmc(pos, N_BURN, progress=True)
    sampler.reset()

    print("Running production...")
    sampler.run_mcmc(pos, N_PROD, progress=True)

    # 3. Analyze Results
    samples = sampler.get_chain(flat=True)
    best_fit = np.median(samples, axis=0)
    stds = np.std(samples, axis=0)
    
    print("\n--- Final MCMC Best Fit ---")
    with open("mcmc_results.txt", "w") as f:
        for i, label in enumerate(LABELS):
            res_str = f"{label}: {best_fit[i]:.4f} +/- {stds[i]:.4f}"
            print(res_str)
            f.write(res_str + "\n")

    fig = corner.corner(samples, labels=LABELS, truths=best_fit)
    fig.savefig("mcmc_dynamic_radius.png")
    print("\nCorner plot saved to 'mcmc_dynamic_radius.png'")