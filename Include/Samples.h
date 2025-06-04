#ifndef SAMPLES_INCLUDED
#define SAMPLES_INCLUDED

#include <vector>
#include "Misha/Miscellany.h"
#include "Misha/Geometry.h"
#include "Misha/Ply.h"
#include "Misha/RegularGrid.h"
#include "Misha/PlyVertexData.h"

namespace MishaK
{
	namespace Samples
	{
		// Gets uniformly random samples from a simplicial mesh
		template< typename Real , unsigned int Dim , unsigned int K , typename Index >
		std::vector< Point< Real , Dim > > GetPositions( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< K , Index > > &simplices , unsigned int numSamples );

		template< typename Real , unsigned int Dim , typename Index >
		std::vector< std::pair< Point< Real , Dim > , Point< Real , Dim > > > GetOrientedPositions( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< Dim-1 , Index > > &simplices , unsigned int numSamples );

		template< typename Real , unsigned int Dim >
		std::vector< Point< Real , Dim > > ReadPositions( std::string fileName , unsigned int numSamples=0 );

		template< typename Real , unsigned int Dim >
		std::vector< std::pair< Point< Real , Dim > , Point< Real , Dim > > > ReadOrientedPositions( std::string fileName , unsigned int numSamples=0 );

		//////////////////
		// Consolidated //
		//////////////////

		template< bool HasNormal , typename Real , unsigned int Dim >
		using Sample = std::conditional_t< HasNormal , std::pair< Point< Real , Dim > , Point< Real , Dim > > , Point< Real , Dim > >;

		template< bool HasNormal , typename Real , unsigned int Dim , unsigned int K , typename Index >
		std::vector< Sample< HasNormal , Real , Dim > > GetSamples( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< K , Index > > &simplices , unsigned int numSamples );

		template< bool HasNormal , typename Real , unsigned int Dim >
		std::vector< Sample< HasNormal , Real , Dim > > ReadSamples( std::string fileName , unsigned int numSamples=-1 );


		////////////////////
		// Implementation //
		////////////////////

		template< typename Real , unsigned int Dim , unsigned int K , typename Index >
		std::vector< Point< Real , Dim  > > GetPositions( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< K , Index > > &simplices , unsigned int numSamples )
		{
			return GetSamples< false >( vertices , simplices , numSamples );
		}

		template< typename Real , unsigned int Dim , typename Index >
		std::vector< std::pair< Point< Real , Dim  > , Point< Real , Dim > > > GetOrientedPositions( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< Dim-1 , Index > > &simplices , unsigned int numSamples )
		{
			return GetSamples< true >( vertices , simplices , numSamples );
		}

		template< typename Real , unsigned int Dim >
		std::vector< Point< Real , Dim  > > ReadPositions( std::string fileName , unsigned int numSamples )
		{
			return ReadSamples< false , Real , Dim >( fileName , numSamples );
		}

		template< typename Real , unsigned int Dim >
		std::vector< std::pair< Point< Real , Dim  > , Point< Real , Dim > > > ReadOrientedPositions( std::string fileName , unsigned int numSamples )
		{
			return ReadSamples< true , Real , Dim >( fileName , numSamples );
		}

		template< bool HasNormal , typename Real , unsigned int Dim , unsigned int K , typename Index >
		std::vector< Sample< HasNormal , Real , Dim > > GetSamples( const std::vector< Point< Real , Dim > > &vertices , const std::vector< SimplexIndex< K , Index > > &simplices , unsigned int numSamples )
		{
			static_assert( !HasNormal || K==Dim-1 , "[ERROR] Can only get normals from co-dimension one mesh" );
			static_assert( Dim!=1 , "[ERROR] Cannot sample 0-dimensional simplicial mesh" );

			auto GetSimplex = [&]( unsigned int i ) -> Simplex< Real , Dim , Dim-1 >
				{
					Simplex< Real , Dim , Dim-1 > simplex;
					for( unsigned int d=0 ; d<Dim ; d++ ) simplex[d] = vertices[ simplices[i][d] ];
					return simplex;
				};

			Real totalMeasure;
			std::vector< Real > measureCDF( simplices.size()+1 );
			{
				measureCDF[0] = 0.;
				for( unsigned int i=0 ; i<simplices.size() ; i++ ) measureCDF[i+1] = measureCDF[i] + GetSimplex(i).measure();
				totalMeasure = measureCDF.back();
				for( unsigned i=0 ; i<measureCDF.size() ; i++ ) measureCDF[i] /= totalMeasure;
			}

			// Find the index i\in[ 0 , simplices.size() ) s.t. cdf[i] <= r <= cdf[i+1];
			auto GetSample = [&]( Real r ) -> Sample< HasNormal , Real , Dim >
				{
					// [low,high] represents a valid range over which the length could be found
					unsigned int low = 0 , high = (unsigned int)measureCDF.size();
					// Invariant:
					// -- lengths[low] <= r <= lengths[high]
					while( low+1<high )
					{
						unsigned int mid = ( low + high ) / 2;
						if( r<=measureCDF[mid] ) high = mid;
						else                      low = mid;
					}
					if( r<measureCDF[low] || r>measureCDF[low+1] ) MK_ERROR_OUT( "badness" );

					Simplex< Real , Dim , Dim-1 > simplex = GetSimplex( low );
					Point< Real , Dim > p = simplex.randomSample();

					if constexpr( HasNormal )
					{
						Point< Real , Dim > n;
						Point< Real , Dim > dv[Dim-1];
						for( unsigned int d=0 ; d<Dim-1 ; d++ ) dv[d] = simplex[d+1]-simplex[0];
						n = Point< Real , Dim >::CrossProduct( dv );
						n /= sqrt( Point< Real , Dim >::SquareNorm( n ) );
						return std::pair< Point< Real , Dim > , Point< Real , Dim > >( p , n );
					}
					else return p;
				};

			std::vector< Sample< HasNormal , Real , Dim > > samples( numSamples );
			for( unsigned int i=0 ; i<numSamples ; i++ )
			{
				Sample< HasNormal , Real , Dim > sample = GetSample( Random< Real >() );
				if constexpr( HasNormal ) sample.second *= totalMeasure / numSamples;
				samples[i] = sample;
			}

			return samples;
		}

		template< bool HasNormal , typename Real , unsigned int Dim >
		std::vector< Sample< HasNormal , Real , Dim > > ReadSamples( std::string fileName , unsigned int numSamples )
		{
			std::string ext = GetFileExtension( fileName );
			ext = ToUpper( ext );

			if( numSamples )
			{
				if constexpr( Dim==1 )
				{
					MK_THROW( "Cannot sample 0-dimensional simplicial mesh" );
					return std::vector< Sample< HasNormal , Real , Dim > >();
				}
				else
				{
					if( ext==std::string( "PLY" ) )
					{
						using Factory = VertexFactory::PositionFactory< Real, Dim >;
						using Vertex = typename Factory::VertexType;

						Factory factory;
						std::vector< Vertex > vertices;
						std::vector< SimplexIndex< Dim-1 , unsigned int > > simplices;

						PLY::ReadSimplices< Factory , Dim-1 , unsigned int >( fileName , factory , vertices , simplices , nullptr );
						if( !simplices.size() ) MK_THROW( "Failed to read simplices" );
						return GetSamples< HasNormal >( vertices , simplices , numSamples );
					}
					else
					{
						MK_THROW( "Cannot read mesh: " , fileName );
						return std::vector< Sample< HasNormal , Real , Dim > >();
					}
				}
			}
			else
			{
				std::vector< Sample< HasNormal , Real , Dim > > samples;
				if( ext==std::string( "PLY" ) )
				{
					using Factory = std::conditional_t< HasNormal , VertexFactory::Factory< double, VertexFactory::PositionFactory< double, Dim > , VertexFactory::NormalFactory< double, Dim > > , VertexFactory::PositionFactory< double, Dim > >;
					using Vertex = typename Factory::VertexType;
					std::vector< Vertex > vertices;

					Factory factory;
					bool *readFlags = new bool[ factory.plyReadNum() ];
					PLY::Read< Factory , int >( fileName , factory , vertices , nullptr , nullptr , readFlags );
					bool hasNormals = true;
					if constexpr( HasNormal ) hasNormals = factory.template plyValidReadProperties< 1 >( readFlags );
					delete[] readFlags;
					if constexpr( HasNormal ) if( !hasNormals ) MK_THROW( "Could not read normals: " , fileName );

					samples.resize( vertices.size() );
					for( unsigned int i=0 ; i<vertices.size() ; i++ )
					{
						if constexpr( HasNormal )
						{
							samples[i].first  = vertices[i].template get<0>();
							samples[i].second = vertices[i].template get<1>();;
						}
						else samples[i] = vertices[i];
					}
				}
				else
				{
					FILE *fp = fopen( fileName.c_str() , "r" );
					if( !fp ) MK_ERROR_OUT( "Failed to open file for readings: " , fileName );
					while( true )
					{
						float f;
						Sample< HasNormal , Real , Dim > sample;

						bool readPoint = true;
						for( unsigned int d=0 ; d<Dim ; d++ )
						{
							readPoint &= fscanf( fp , " %f" , &f )==1;
							if constexpr( HasNormal ) sample.first[d] = f;
							else sample[d] = f;
						}
						if constexpr( HasNormal )
						{
							for( unsigned int d=0 ; d<Dim ; d++ )
							{
								readPoint &= fscanf( fp , " %f" , &f )==1;
								sample.second[d] = f;
							}
						}
						if( readPoint ) samples.push_back( sample );
						else break;
					}
					fclose( fp );
				}
				return samples;
			}
		}
	}
}
#endif // SAMPLES_INCLUDED