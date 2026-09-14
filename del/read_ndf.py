"""
Simple NDF reader using HDS format (Starlink NDF files are HDS containers).
This reads the DATA array directly from the .sdf file.
"""
import numpy as np
import struct
import os

def read_ndf_simple(ndf_path):
    """
    Very basic NDF reader - reads just the DATA array.
    This is a workaround when Starlink tools are not available.
    """
    # Add .sdf extension if not present
    if not ndf_path.endswith('.sdf'):
        sdf_path = ndf_path + '.sdf'
    else:
        sdf_path = ndf_path
    
    if not os.path.exists(sdf_path):
        raise FileNotFoundError(f"NDF file not found: {sdf_path}")
    
    # Try to read as binary and extract float data
    with open(sdf_path, 'rb') as f:
        content = f.read()
    
    # Look for DATA component - this is a hack but might work
    # HDS files have a specific structure, look for _REAL or _DOUBLE arrays
    try:
        # Try to find where numeric data starts (heuristic)
        # Skip header (usually first ~1KB contains metadata)
        header_size = min(2048, len(content) // 4)
        data_bytes = content[header_size:]
        
        # Try reading as float32 array
        data = np.frombuffer(data_bytes, dtype=np.float32)
        
        # Reshape to 2D (estimate dimensions from data size)
        # Common sizes: 64x64, 128x128, 256x256, etc.
        size = len(data)
        dim = int(np.sqrt(size))
        
        # Try common dimensions
        for test_dim in [64, 128, 256, 512]:
            if test_dim * test_dim <= size:
                dim = test_dim
        
        data_2d = data[:dim*dim].reshape((dim, dim))
        
        return data_2d
        
    except Exception as e:
        raise ValueError(f"Could not parse NDF file: {e}. Please convert to FITS manually.")

if __name__ == "__main__":
    # Test reading
    try:
        data = read_ndf_simple("group1/tmp/m20251129_00078_01_backoff")
        print(f"Successfully read NDF: shape={data.shape}, dtype={data.dtype}")
        print(f"Data range: [{np.nanmin(data):.3e}, {np.nanmax(data):.3e}]")
    except Exception as e:
        print(f"Failed to read NDF: {e}")
