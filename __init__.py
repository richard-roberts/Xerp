from . import affine
from . import interpolation

try:
	reload(affine)
	reload(interpolation)
except:
	print("reload failed (probably not maya)")

LeeColors = affine.LeeColors
LeeAffine = affine.LeeAffine
ScatteredDataInterpolation = interpolation.ScatteredDataInterpolation