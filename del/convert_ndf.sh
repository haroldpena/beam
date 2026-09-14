#!/bin/bash
# Convert NDF to FITS using Starlink tools

# Source Starlink environment (tcsh style)
source /Users/haroldpena/starlink-2025A/star-2025A/etc/profile

# Convert the NDF file
echo "Converting NDF to FITS..."
ndf2fits group1/tmp/m20251129_00078_01_backoff group1/tmp/m20251129_00078_01_backoff.fits

echo "Done! FITS file created at: group1/tmp/m20251129_00078_01_backoff.fits"
