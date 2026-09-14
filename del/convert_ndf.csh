#!/bin/tcsh
# Convert NDF to FITS using Starlink CONVERT package

# Source Starlink environment
source /Users/haroldpena/starlink-2025A/star-2025A/etc/cshrc
source /Users/haroldpena/starlink-2025A/star-2025A/etc/login

# Initialize CONVERT package
convert

# Convert NDF to FITS
ndf2fits in=group1/tmp/m20251129_00078_01_backoff out=group1/tmp/m20251129_00078_01_backoff.fits

echo "Conversion complete!"
ls -lh group1/tmp/m20251129_00078_01_backoff.fits
