#!/usr/bin/env python
"""Convert NDF to FITS using Starlink Python bindings."""
import sys
sys.path.insert(0, '/Users/haroldpena/starlink-2025A/star-2025A/lib/python')

try:
    from starlink import convert
    
    ndf_file = "group1/tmp/m20251129_00078_01_backoff"
    fits_file = "group1/tmp/m20251129_00078_01_backoff.fits"
    
    print(f"Converting {ndf_file} to {fits_file}...")
    convert.ndf2fits(ndf_file, fits_file)
    print("Conversion complete!")
    
except ImportError as e:
    print(f"Starlink Python bindings not available: {e}")
    print("Please convert manually using Starlink command line tools.")
