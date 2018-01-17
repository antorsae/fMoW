from osgeo import gdal
import numpy as np
import iterm
import sys

f= sys.argv[1] # '/home/antor/sat/fMoW-full/train/airport/airport_228/airport_228_4_ms.tif' #test/0001826/0001826_0_ms.tif'
ds = gdal.Open(f)
print ds.GetMetadata()

def to_uint8_raster(a):
	_min, _max = a.min(), a.max()
	print(a.shape, _min, _max)
	return (255. * (a - _min) / (_max - _min) ).astype(np.uint8)

# loop through each band
for bi in range(ds.RasterCount):
    band = ds.GetRasterBand(bi + 1)
    # Read this band into a 2D NumPy array
    ar = band.ReadAsArray()
    #iterm.show_image(to_uint8_raster(ar))
    print('Band %d has type %s and shape (%d, %d)'% (bi + 1, ar.dtype, ar.shape[0], ar.shape[1]))
    raw = ar.tostring()
