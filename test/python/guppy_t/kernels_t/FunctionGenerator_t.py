#!/usr/bin/env python
# encoding: utf-8

import unittest
from guppy.kernels.FunctionGenerator import FunctionGenerator
import numpy
import random

class FunctionGenerator_t(unittest.TestCase):

	def setUp(self):

		#Request a 32*32 grid of threads to run kernel on

		self.globalsize = (32,32)

		#Define limits of grid. Here x,y,z are integer divisions between 0-9, which defines a grid of 1000 points (0,0,0) to (9,9,9)
		#N-Dimensions are parsed as an array of shape (N,3). Each dimension is of the form [low, high, no. divisions]
		
                self.grid = numpy.array([[0,10,10],[0,10,10],[0,10,10]])
		
		# Note that: This simple unit test only works with integer divisions, so that dimension[index]=index.
		# If you change the number of divisions, you will need to calculate the values within the dimension to ensure the
	        # correct checksum
		# 
		#	Example:
		#
		#	x0=y0 = [0,10,10] has 10 divisions 0-> 9 so that checksum x0[i]+y0[j] = i+j
		#	x1=y1 = [0,10,20] has 20 divisions 0-> 19 so that checksum x1[i]+y1[j] != i+j , will fail the unit test		 
		#
		# The use of integer division is purely to provide a simple example of how the function generator works. To test
		# functionality for non-integer divisions, you will need to calculate the x,y,z values manually to build a checksum 

		#Three arbitrary values between 0..9, to create a checksum to compare local result with gpu result
		self.x0 = 3
		self.x1 = 8
		self.x2 = 2

	def test_add3D(self):
		print "\naddition test..."		
		self.function_string = 'dim[0]+dim[1]+dim[2];'
		FG = FunctionGenerator(self.function_string, 3)	
		add3Dresult = FG(globalsize=self.globalsize, grid=self.grid)
		#Check addition result
		checksum = self.x0+self.x1+self.x2 
		self.assertEquals(checksum, add3Dresult[self.x0,self.x1,self.x2])

	def test_sub3D(self):
		print "\nsubtraction test..."
		self.function_string = 'dim[0]-dim[1]-dim[2];'
                FG = FunctionGenerator(self.function_string, 3)
                sub3Dresult = FG(globalsize=self.globalsize, grid=self.grid)
                #Check subtraction result
		checksum = self.x0-self.x1-self.x2
		self.assertEquals(checksum, sub3Dresult[self.x0,self.x1,self.x2])

	def test_mul3D(self):
		print "\nmultiplication test..."
		self.function_string = 'dim[0]*dim[1]*dim[2];'
                FG = FunctionGenerator(self.function_string, 3)
                mul3Dresult = FG(globalsize=self.globalsize, grid=self.grid)
                #Check multiplication result
		checksum = self.x0*self.x1*self.x2
		self.assertEquals(checksum, mul3Dresult[self.x0,self.x1,self.x2])

	def test_div3D(self):
		
                print "\ndivision test..."                     
                self.function_string = 'dim[0]/dim[1]/dim[2];'
                FG = FunctionGenerator(self.function_string, 3)
                div3Dresult = FG(globalsize=self.globalsize, grid=self.grid)
                #Check multiplication result
                checksum = float(self.x0)/float(self.x1)/float(self.x2)
                #Typically fails due to floating point errors?
		self.assertEquals(float(checksum), div3Dresult[self.x0,self.x1,self.x2])

	def test_exp3D(self):
		print "\nexponential test..."
		self.function_string = 'exp(dim[0]+dim[1]);'
		FG = FunctionGenerator(self.function_string, 3)
                exp3Dresult = FG(globalsize=self.globalsize, grid=self.grid)
                #Check multiplication result
		# Note: when using numpy.exp, ensure that the checksum is same dtype as in
		# the kernel class i.e. float32.....numpy defaults to float64 otherwise and will fail test

                checksum = numpy.array([float(self.x0)+float(self.x1)], dtype =numpy.float32) 
                self.assertEquals(numpy.exp(checksum)[0], exp3Dresult[self.x0,self.x1,self.x2])

if __name__ == '__main__':
	unittest.main()
