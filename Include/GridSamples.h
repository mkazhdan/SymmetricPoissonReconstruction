/*
Copyright (c) 2025, Michael Kazhdan, Sing-Chun Lee, Marc Alexa, and Maximilian Kohlbrenner
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution. 

Neither the name of the Johns Hopkins University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission. 

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES 
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
*/

#ifndef GRID_SAMPLES_INCLUDED
#define GRID_SAMPLES_INCLUDED

#include <vector>
#include <Eigen/Eigenvalues>
#include "Misha/Miscellany.h"
#include "Misha/Geometry.h"
#include "Misha/RegularGrid.h"
#include "Misha/Algebra.h"
#include "Misha/Streams.h"
#include "Misha/MultiThreading.h"
#include "Hat.h"
#include "Misha/RegularTree.h"
#include "OrderedSamples.h"

namespace MishaK
{
	namespace GridSamples
	{
		template< unsigned int Degree=(unsigned int)-1 >struct BSpline;

		// Run-time implementation of B-spline weight evaluation
		template<>
		struct BSpline< (unsigned int)-1 >
		{
			// Assuming x \in [0,1], evaluates the supported B-splines
			static void Evaluate( unsigned int degree , double x , double values[/*degree+1*/] );
		};

		// Templated implementation of B-spline weight evaluation
		template< unsigned int Degree >
		struct BSpline
		{
			// Assuming x \in [0,1], evaluates the supported B-splines
			static void Evaluate( double x , double values[Degree+1] );
		};

		// A pure abstract class that supports evaluating the measure, noise, and depth at a point in the unit cube.
		template< unsigned int Dim > struct Estimator
		{
			// The measure per unit volume at a given position
			virtual double measure( Point< double , Dim > , unsigned int thread ) const = 0;
			// The estimate of noise at a given position
			virtual double noise( Point< double , Dim > , unsigned int thread ) const = 0;
			// The measure of the sample's depth at a given position
			virtual double depth( Point< double , Dim > , unsigned int thread ) const = 0;
		};

		// A struct describing the distribution of samples
		template< unsigned int Dim >
		struct DensityInfo : public VectorSpace< double , DensityInfo< Dim > >
		{
			double weightSum;
			Point< double , Dim > positionSum;
			DensityInfo( void );
			DensityInfo( Point< double , Dim > p , double weight );
			//////////////////////////
			// Vector space methods //
			void Add  ( const DensityInfo &ni );
			void Scale( double s );
			//////////////////////////

			double noise( unsigned int coDim ) const;
			double measure( void ) const;
		};

		template< unsigned int Dim >
		struct DensityAndNoiseInfo : public VectorSpace< double , DensityAndNoiseInfo< Dim > >
		{
			double weightSum;
			Point< double , Dim > positionSum;
			SquareMatrix< double , Dim > covarianceSum;
			DensityAndNoiseInfo( void );
			DensityAndNoiseInfo( Point< double , Dim > p , double weight );
			//////////////////////////
			// Vector space methods //
			void Add  ( const DensityAndNoiseInfo &ni );
			void Scale( double s );
			//////////////////////////

			double noise( unsigned int coDim ) const;
			double measure( void ) const;
		};

		// A struct for splatting a sample into a grid
		template< unsigned int Dim >
		struct Splatter
		{
			Splatter( unsigned int depth , unsigned int bSplineDegree , bool unitKernel );
			~Splatter( void );

			template< typename DataType >
			void operator()( DataType *g , std::pair< Point< double , Dim > , DataType > sample );

			// Splats the values into the corners of the cell containg the sample using the B-spline weights
			template< typename DataType , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( size_t idx ) */ >
			static RegularGrid< Dim , DataType > SplatSamples( unsigned int bSplineDegree , unsigned int depth , SampleFunctor F , size_t sampleNum , DataType zeroValue , bool unitKernel );

			// Splats the values into the corners of the cell containg the sample using multi-linear weights
			template< typename DataType , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( size_t idx ) */ >
			static void SplatSamples( unsigned int bSplineDegree , RegularGrid< Dim , DataType > &g , SampleFunctor F , size_t sampleNum , bool unitKernel );

			// Splats the values into the corners of the cell containg the sample using multi-linear weights
			template< typename DataType , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( size_t idx ) */ >
			static void SplatSamples_parallel( unsigned int bSplineDegree , RegularGrid< Dim , DataType > &g , SampleFunctor F , size_t sampleNum , DataType zeroValue , bool unitKernel );

		protected:
			double *_values;
			Hat::ScalarFunctions< Dim > _scalars;
			int _bSplineStart;
			unsigned int _bSplineDegree , _res;
			Hat::Range< Dim > _range;
			double _kernelScale;
		};

		template< unsigned int Dim , typename DataType >
		struct TreeSplatter
		{
			using Node = RegularTreeNode< Dim , DataType , unsigned short >;

			TreeSplatter( unsigned int maxDepth );
			TreeSplatter( BinaryStream &stream );

			template< unsigned int KernelRadius , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( size_t idx ) > */ , typename DepthWeightFunctor /* = std::function< double > ( unsigned int depth ) > */ >
			void addSamples( SampleFunctor && Sample , const OrderedSampler< Dim > &orderedSampler , DepthWeightFunctor && DepthWeight , bool mergeSamples );

			template< unsigned int KernelRadius , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( size_t idx ) > */ , typename SampleDepthFunctor /* = std::function< double ( size_t idx ) > */ , typename DepthWeightFunctor /* = std::function< double > ( unsigned int depth ) > */ >
			void addSamples( SampleFunctor && Sample , const OrderedSampler< Dim > &orderedSampler , SampleDepthFunctor && SampleDepth , DepthWeightFunctor && DepthWeight , bool mergeSamples );

			// Returns the density and noise information at the specified point and the specified depth
			DataType operator()( Point< double , Dim > p , unsigned int depth , unsigned int thread ) const;

			// Returns the density and noise information at the specified point, summed across all depths
			DataType operator()( Point< double , Dim > p , unsigned int thread ) const;

			// Processes the data in the tree
			template< typename DataFunctor /* = std::function< void ( unsigned int , Hat::Index< Dim > , const DataType & ) > */ >
			void process( DataFunctor && F ) const;

			// Processes the data in the tree
			template< typename DataFunctor /* = std::function< void ( unsigned int , Hat::Index< Dim > , DataType & ) > */ >
			void process( DataFunctor && F );

			// Write the data in the tree to a stream
			void write( BinaryStream &stream , bool serialize ) const;

			// Read the data in the tree to a stream
			void read( BinaryStream &stream );

		protected:
			template< unsigned int KernelRadius >
			using NeighborKey = typename Node::template NeighborKey< ParameterPack::IsotropicUIntPack< Dim , KernelRadius > , ParameterPack::IsotropicUIntPack< Dim , KernelRadius+1 > >;
			template< unsigned int KernelRadius >
			using ConstNeighborKey = typename Node::template ConstNeighborKey< ParameterPack::IsotropicUIntPack< Dim , KernelRadius > , ParameterPack::IsotropicUIntPack< Dim , KernelRadius+1 > >;

			template< unsigned int KernelRadius >
			struct _InsertionData
			{
				static const unsigned int BSplineDegree = 2*KernelRadius + 1;
				NeighborKey< KernelRadius > nKey;
				double bSplineValues[ Dim ][ BSplineDegree+1 ];
				_InsertionData( unsigned int maxDepth ){ nKey.set( maxDepth+1 ); }
			};

			Node _root , *_spaceRoot;
			unsigned int _maxDepth;
			std::vector< ConstNeighborKey< 0 > > _nKeys;

			template< unsigned int KernelRadius , typename DepthWeightFunctor /* = std::function< Data ( size_t idx ) > */ >
			void _addSample( _InsertionData< KernelRadius > &iData , Point< double , Dim > p , DataType data , DepthWeightFunctor && F );

			template< unsigned int KernelRadius , typename DepthWeightFunctor /* = std::function< Data ( size_t idx ) > */ >
			void _addSample( _InsertionData< KernelRadius > &iData , double depth , Point< double , Dim > p , DataType data , DepthWeightFunctor && F );

			template< unsigned int KernelRadius >
			void _addSample( _InsertionData< KernelRadius > &iData , unsigned int depth , Point< double , Dim > p , DataType data );

			template< unsigned int KernelRadius , typename PositionFunctor /* = std::function< Point< double , Dim > ( size_t idx ) > */  , typename DataFunctor /* = std::function< Data ( size_t idx ) > */ , typename DepthWeightFunctor /* = std::function< double ( size_t idx ) > */ >
			void _addSamples( NeighborKey< KernelRadius > &nKey , size_t sz , PositionFunctor && PositionF , DataFunctor && DataF , DepthWeightFunctor && DepthF );
		};

		// A struct for splatting a sample from a (Dim-CoDim)-dimensional manifold, living in Dim-dimensional Euclidean space into a grid
		template< unsigned int Dim , unsigned int CoDim >
		struct SingleEstimator : public RegularGrid< Dim , DensityAndNoiseInfo< Dim > > , Estimator< Dim >
		{
			SingleEstimator( unsigned int depth );
			unsigned int depth( void ) const;
			double        measure( Point< double , Dim > p , unsigned int thread ) const;
			double          noise( Point< double , Dim > p , unsigned int thread ) const;
			double          depth( Point< double , Dim > p , unsigned int thread ) const;
			DensityAndNoiseInfo< Dim > operator()( Point< double , Dim > p , unsigned int thread ) const;
			double samplesPerCell( Point< double , Dim > p , unsigned int thread ) const;

			template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( size_t idx ) */ >
			static SingleEstimator Get( unsigned int kernelRadius , unsigned int depth , SampleFunctor F , size_t sampleNum );

			template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( size_t idx ) */ >
			static void Set( SingleEstimator &estimator , unsigned int kernelRadius , SampleFunctor F , size_t sampleNum );

		protected:
			unsigned int _depth;
		};

		template< unsigned int Dim , unsigned int CoDim >
		struct MultiEstimator : public Estimator< Dim >
		{
			MultiEstimator( unsigned int depth , double samplesPerCell );
			DensityAndNoiseInfo< Dim > operator()( Point< double , Dim > p , unsigned int thread ) const;
			double   depth( Point< double , Dim > p , unsigned int thread ) const;
			double measure( Point< double , Dim > p , unsigned int thread ) const;
			double   noise( Point< double , Dim > p , unsigned int thread ) const;
			double samplesPerCell( Point< double , Dim > p , unsigned int thread ) const;

			template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( size_t idx ) */ >
			static MultiEstimator Get( unsigned int kernelRadius , unsigned int depth , SampleFunctor F , size_t sampleNum , double samplesPerCell=pow(2.,Dim-CoDim) );
		protected:
			std::vector< SingleEstimator< Dim , CoDim > > _info;
			double _samplesPerCell;
		};

		template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
		struct TreeEstimator : public Estimator< Dim > , TreeSplatter< Dim , std::conditional_t< HasNoise , DensityAndNoiseInfo< Dim > , DensityInfo< Dim > > >
		{
			using Info = std::conditional_t< HasNoise , DensityAndNoiseInfo< Dim > , DensityInfo< Dim > >;
			using Node = typename TreeSplatter< Dim , Info >::Node;

			TreeEstimator( void );
			TreeEstimator( BinaryStream &stream );

			template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( size_t idx ) */ >
			TreeEstimator( unsigned int kernelRadius , unsigned int maxDepth , SampleFunctor && F , const OrderedSampler< Dim > &orderedSampler , bool mergeSamples , double samplesPerCell=pow(2.,Dim-CoDim) );

			// Returns the density and noise information at the specified point
			Info operator()( Point< double , Dim > p , unsigned int thread ) const;

			// The measure per unit volume at a given position
			double measure( Point< double , Dim > , unsigned int thread ) const;
			// The estimate of noise at a given position
			double noise( Point< double , Dim > , unsigned int thread ) const;
			// The measure of the sample's depth at a given position
			double depth( Point< double , Dim > , unsigned int thread ) const;

			// The estimate of the number of samples per cell at the point p, measured with respect to a cell at the finest depth
			double samplesPerCell( Point< double , Dim > p , unsigned int thread ) const;

			// Write the data in the tree to a stream
			void write( BinaryStream &stream , bool serialize ) const;

			// Read the data in the tree to a stream
			void read( BinaryStream &stream );

		protected:
			template< unsigned int KernelRadius >
			using NeighborKey = typename Node::template NeighborKey< ParameterPack::IsotropicUIntPack< Dim , KernelRadius > , ParameterPack::IsotropicUIntPack< Dim , KernelRadius+1 > >;
			template< unsigned int KernelRadius >
			using ConstNeighborKey = typename Node::template ConstNeighborKey< ParameterPack::IsotropicUIntPack< Dim , KernelRadius > , ParameterPack::IsotropicUIntPack< Dim , KernelRadius+1 > >;

			double _targetSamplesPerCell;

			using TreeSplatter< Dim , Info >::_maxDepth;
			using TreeSplatter< Dim , Info >::_nKeys;
			using TreeSplatter< Dim , Info >::_spaceRoot;

			Info _info( ConstNeighborKey< 0 > &nKey , unsigned int depth , Point< double , Dim > p ) const;
			double _samplesPerCell( ConstNeighborKey< 0 > &nKey , unsigned int depth , Point< double , Dim > p ) const;
			double _depth( ConstNeighborKey< 0 > &nKey , unsigned int _maxDepth , Point< double , Dim > p ) const;
		};
#include "GridSamples.inl"
	}
}

#endif // GRID_SAMPLES_INCLUDED