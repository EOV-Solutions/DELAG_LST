import rasterio

def read_tif_metadata(tif_path):
    """
    Read and print metadata from a GeoTIFF file using rasterio.
    
    Args:
        tif_path (str): Path to the GeoTIFF file
        
    Returns:
        dict: Dictionary containing the metadata
    """
    try:
        with rasterio.open(tif_path) as src:
            # Get basic metadata
            metadata = {
                'driver': src.driver,
                'dtype': src.dtypes[0],
                'nodata': src.nodata,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds
            }
            
            # Print metadata
            print(f"\nMetadata for {tif_path}:")
            print(f"Driver: {metadata['driver']}")
            print(f"Data type: {metadata['dtype']}")
            print(f"No data value: {metadata['nodata']}")
            print(f"Width: {metadata['width']}")
            print(f"Height: {metadata['height']}")
            print(f"Number of bands: {metadata['count']}")
            print(f"Coordinate Reference System: {metadata['crs']}")
            print(f"Transform: {metadata['transform']}")
            print(f"Bounds: {metadata['bounds']}")
            print(f"Resolution: {metadata['transform'][0]} x {abs(metadata['transform'][4])} meters")
            
            return metadata
            
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return None

# Example usage:
tif_path = "/mnt/ssd1tb/code/nhatvm/DELAG/DELAG_LST/KhanhXuan_BuonMaThuot_DakLak/ndvi_infer/ndvi_2023-01-01.tif"
metadata = read_tif_metadata(tif_path)
