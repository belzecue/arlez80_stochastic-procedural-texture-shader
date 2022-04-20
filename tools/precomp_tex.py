#
# 確率論的テクスチャ用事前計算 / Ported by あるる（きのもと 結衣） @arlez80
# Pre Computations for Procedural Stochastic Textures / Ported by Yui Kinomoto @arlez80
#
# MIT License
#
#	References:
#		https://eheitzresearch.wordpress.com/722-2/
#				High-Performance By-Example Noise using a Histogram-Preserving Blending Operator
#		https://eheitzresearch.wordpress.com/738-2/
# 			"Procedural Stochastic Textures by Tiling and Blending"
#

import math
import cv2
import numpy as np
import statistics

GAUSSIAN_AVERAGE = 0.5
GAUSSIAN_STD = 0.166666
LUT_WITH = 128

class Precomputations:
	input = None
	input_decorrelated = None
	t_input = None
	t_inv = None
	color_space_vector1 = None
	color_space_vector2 = None
	color_space_vector3 = None
	color_space_origin = None

	def exec( self, path ):
		print( "Start" )
		self.input = ( cv2.imread( path ) / 255.0 ).astype( np.float64 )[:,:,::-1]
		self.input_decorrelated = np.zeros_like( self.input )
		self.t_input = np.zeros_like( self.input )
		self.t_inv = np.zeros( ( 1, LUT_WITH, 3 ) )

		# Section 1.4 Improvement: usign a decorrelated color space
		print( "	--> Generating Decorrelate Color Space" )
		self._decorrelate_color_space( )

		# Section 1.3.2 Applying the histogram transformation T on the input
		print( "	--> Generating T Input" )
		self._compute_t_input( )

		# Section 1.3.3 Precomputing the inverse histogram transformation T^{-1}
		print( "	--> Generating Inv T" )
		self._compute_inv_t( )

		# Section 1.5 Improvement: prefiltering the loop-up table
		# TODO: implement

		self.input = self.input[:,:,::-1]
		self.input_decorrelated = self.input_decorrelated[:,:,::-1]
		self.t_input = self.t_input[:,:,::-1]
		self.t_inv = self.t_inv[:,:,::-1]
		print( "Finished" )

	def _decorrelate_color_space( self ):
		eigen_vectors = self._compute_eigen_vectors( )

		# Rotate to eigen vector space and
		# TODO: 高速化
		for y in range( self.input.shape[0] ):
			input_line = self.input[y]
			for x in range( self.input.shape[1] ):
				v = input_line[x]
				self.input_decorrelated[y][x] = (
					np.dot( v, eigen_vectors[0] )
				,	np.dot( v, eigen_vectors[1] )
				,	np.dot( v, eigen_vectors[2] )
				)

		# Compute ranges of the new color space
		color_space_ranges = np.array((
			( np.amin( self.input_decorrelated[:,:,0] ), np.amax( self.input_decorrelated[:,:,0] ) )
		,	( np.amin( self.input_decorrelated[:,:,1] ), np.amax( self.input_decorrelated[:,:,1] ) )
		,	( np.amin( self.input_decorrelated[:,:,2] ), np.amax( self.input_decorrelated[:,:,2] ) )
		))

		# Remap range to [0, 1]
		self.input_decorrelated = ( self.input_decorrelated - color_space_ranges[:,0] ) / ( color_space_ranges[:,1] - color_space_ranges[:,0] )

		self.color_space_origin = np.array((
			color_space_ranges[0][0] * eigen_vectors[0][0] + color_space_ranges[1][0] * eigen_vectors[1][0] + color_space_ranges[2][0] * eigen_vectors[2][0]
		,	color_space_ranges[0][0] * eigen_vectors[0][1] + color_space_ranges[1][0] * eigen_vectors[1][1] + color_space_ranges[2][0] * eigen_vectors[2][1]
		,	color_space_ranges[0][0] * eigen_vectors[0][2] + color_space_ranges[1][0] * eigen_vectors[1][2] + color_space_ranges[2][0] * eigen_vectors[2][2]
		))
		
		self.color_space_vector1 = eigen_vectors[0] * ( color_space_ranges[0][1] - color_space_ranges[0][0] )
		self.color_space_vector2 = eigen_vectors[1] * ( color_space_ranges[1][1] - color_space_ranges[1][0] )
		self.color_space_vector3 = eigen_vectors[2] * ( color_space_ranges[2][1] - color_space_ranges[2][0] )

	def _compute_eigen_vectors( self ):
		size = float( self.input.shape[0] * self.input.shape[1] )
		c = np.sum( self.input, axis=(0, 1) ) / size
		c_dot = np.sum( self.input * self.input, axis=(0, 1) ) / size
		rg = np.sum( self.input[:,:,0] * self.input[:,:,1], axis=(0, 1) ) / size
		rb = np.sum( self.input[:,:,0] * self.input[:,:,2], axis=(0, 1) ) / size
		gb = np.sum( self.input[:,:,1] * self.input[:,:,2], axis=(0, 1) ) / size

		_, _, eigen_vectors = cv2.eigen(np.array((
			( c_dot[0] - c[0]*c[0], rg - c[0]*c[1], rb - c[0]*c[2] )
		,	( rg - c[0]*c[1], c_dot[1] - c[1]*c[1], gb - c[1]*c[2] )
		,	( rb - c[0]*c[2], gb - c[1]*c[2], c_dot[2] - c[2]*c[2] )
		)))
		return np.array((
			eigen_vectors[0]
		,	eigen_vectors[2]
		,	eigen_vectors[1]
		))

	def _compute_t_input( self ):
		input_values = [[],[],[]]
		for y in range( self.input_decorrelated.shape[0] ):
			for x in range( self.input_decorrelated.shape[1] ):
				v = self.input_decorrelated[y][x]

				for ch in range( 3 ):
					input_values[ch].append((x,y,v[ch]))

		sorted_input_values = []
		for ch in range( 3 ):
			v = np.array( input_values[ch] )
			sorted_input_values.append( v[np.argsort(v[:, 2])] )

		size = len( sorted_input_values[0] )
		nd = statistics.NormalDist( GAUSSIAN_AVERAGE, GAUSSIAN_STD )
		for i in range( size ):
			u = ( i + 0.5 ) / size
			g = nd.inv_cdf( u )

			for ch in range( 3 ):
				x, y, _ = sorted_input_values[ch][i]
				self.t_input[int(y)][int(x)][ch] = g

	def _compute_inv_t( self ):
		input_values = [[],[],[]]

		for y in range( self.input_decorrelated.shape[0] ):
			for x in range( self.input_decorrelated.shape[1] ):
				v = self.input_decorrelated[y][x]

				for ch in range( 3 ):
					input_values[ch].append( v[ch] )

		sorted_input_values = [ np.sort( np.array( input_values[i] ) ) for i in range( 3 ) ]
		size_input_values = len( sorted_input_values[0] )
		lut_size = self.t_inv.shape[1]
		nd = statistics.NormalDist( GAUSSIAN_AVERAGE, GAUSSIAN_STD )
		for i in range( lut_size ):
			g = ( i + 0.5 ) / lut_size
			u = nd.cdf( g )
			index = int( math.floor( u * size_input_values ) )
			for ch in range( 3 ):
				self.t_inv[0][i][ch] = sorted_input_values[ch][index]

def _main( ):
	import argparse
	import os
	import json

	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", help="Input texture file path", required=True)
	parser.add_argument("-ot", "--output-t-input", help="Output TInput file path")
	parser.add_argument("-ol", "--output-lut", help="Output LUT file path")
	parser.add_argument("-oc", "--output-color-space", help="Output Color Spaces file path")
	parser.add_argument("--test", help="Show the converted image (no save)", action="store_true")
	args = parser.parse_args( )

	st = Precomputations( )
	st.exec( args.input )

	print( "Color Space Origin: ", st.color_space_origin )
	print( "Color Space Vector1: ", st.color_space_vector1 )
	print( "Color Space Vector2: ", st.color_space_vector2 )
	print( "Color Space Vector3: ", st.color_space_vector3 )

	if args.test:
		cv2.imshow( "input", st.input )
		cv2.imshow( "input_decorrelated", st.input_decorrelated )
		cv2.imshow( "t_input", st.t_input )
		cv2.imshow( "t_inv", st.t_inv )
		cv2.waitKey( 0 )
		cv2.destroyAllWindows( )
	else:
		base_file_name = os.path.splitext( os.path.basename( args.input ) )[0]
		t_input_path = base_file_name + "_t_input.png"
		if args.output_t_input:
			t_input_path = args.output_t_input
		lut_path = base_file_name + "_lut.png"
		if args.output_lut:
			lut_path = args.output_lut
		color_space_path = base_file_name + "_color_space.json"
		if args.output_color_space:
			color_space_path = args.output_color_space

		cv2.imwrite( t_input_path, np.clip( st.t_input * 255, a_min=0, a_max=255 ).astype( np.uint8 ) )
		cv2.imwrite( lut_path, np.clip( st.t_inv * 255, a_min=0, a_max=255 ).astype( np.uint8 ) )
		with open( color_space_path, 'w' ) as f:
			json.dump({
				"color_space_origin": st.color_space_origin.tolist( )
			,	"color_space_vector1": st.color_space_vector1.tolist( )
			,	"color_space_vector2": st.color_space_vector2.tolist( )
			,	"color_space_vector3": st.color_space_vector3.tolist( )
			}, f)

if __name__ == "__main__":
	_main( )
