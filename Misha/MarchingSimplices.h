/*
Copyright (c) 2023, Michael Kazhdan
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
#ifndef MARCHING_SIMPLICES_INCLUDED
#define MARCHING_SIMPLICES_INCLUDED

#include <vector>
#include <unordered_set>
#include <set>
#include <map>
#include "Eigen/Dense"
#include "Geometry.h"
#include "HyperCubeSimplices.h"

namespace MishaK
{
	namespace MarchingSimplices
	{
		// A structure for storing a simplicial mesh with Data types at vertices
		template< unsigned int Dim , typename Index , typename Data >
		struct SimplicialMesh
		{
			std::vector< SimplexIndex< Dim , Index > > simplexIndices;
			std::vector< Data > vertices;
		};

		template< unsigned int K , typename Index >
		std::vector< std::vector< size_t > > FaceAdjacentConnectedComponents( const std::vector< SimplexIndex< K , Index > > &simplexIndices );

		template< unsigned int Dim , typename Index >
		bool IsOriented( const std::vector< SimplexIndex< Dim , Index > > &simplexIndices );

		template< unsigned int K , typename Index >
		unsigned int RemoveDuplicates( std::vector< SimplexIndex< K , Index > > &simplexIndices );

		template< unsigned int K , typename Index >
		void Orient( std::vector< SimplexIndex< K , Index > > &simplexIndices );

		// Functionality for extracting the (Dim-CoDim)-dimensional level-set of a Dim-dimensional simplicial mesh
		// with CoDim-dimensional values associated (via the DataFunctor) to the vertices
		template< typename DataReal , unsigned CoDim , unsigned int Dim , typename Index , typename Data , typename PositionFunction /* = std::function< Point< double , CoDim > ( unsigned int ) > */ > 
		SimplicialMesh< Dim - CoDim , Index , Data > LevelSet( const SimplicialMesh< Dim , Index , Data > &sMesh , PositionFunction F , Point< double , CoDim > isoValue );

		// Functionality for obtaining a simplicial mesh triangulating the corners of a regular grid (possibly by adding cell centers)
		template< unsigned int Dim >
		SimplicialMesh< Dim , unsigned int , Point< double , Dim > > RegularGridTriangulation( unsigned int res , bool orient=true , bool addCenters=false );

		template< unsigned int Dim , typename IndexFunctor /* = std::function< Point< unsigned int , Dim > ( size_t ) > */ >
		SimplicialMesh< Dim , unsigned int , Point< double , Dim > > RegularSubGridTriangulation( size_t eNum , IndexFunctor && F , bool orient=true , bool addCenters=false );

		// Functionality for returning the v-th boundary face of a simplex, with consistent orientation
		template< unsigned int Dim , typename Index >
		SimplexIndex< Dim-1 , Index > OrientedBoundaryFace( SimplexIndex< Dim , Index > si , Index v );

		////////////////////
		// Implementation //
		////////////////////

		template< unsigned int Dim , typename Index >
		SimplexIndex< Dim-1 , Index > OrientedBoundaryFace( SimplexIndex< Dim , Index > si , Index v )
		{
			// Think of the simplex as living in Dim+1 dimensions spanned by vectors {e_0,...,e_Dim}
			// The simplex face {e_{v+1},...,e_{v+Dim}} is properly oriented if...
			SimplexIndex< Dim-1 , Index > _si;
			for( unsigned int d=0 ; d<Dim ; d++ ) _si[d] = si[(v+d+1)%(Dim+1)];
			if( Dim&1 && v&1 ) std::swap( _si[0] , _si[1] );
			return _si;
		}

		template< unsigned int Size , typename Index >
		struct MultiIndex
		{
			MultiIndex( void ){ for( unsigned int i=0 ; i<Size ; i++ ) _indices[i] = 0; }
			MultiIndex( const Index indices[] ){ _init( indices ); }
			MultiIndex(       Index indices[] ){ _init( indices ); }
			template< typename ... Is > MultiIndex( Is ... indices );
			bool operator < ( const MultiIndex &idx ) const;
			bool operator == ( const MultiIndex &idx ) const;
			const Index &operator[] ( unsigned int idx ) const { return _indices[idx]; }
			typedef std::map< MultiIndex , Index > map;
			typedef std::set< MultiIndex         > set;

		protected:
			void _init( const Index indices[] );
			Index _indices[ Size ];
		};

		template< unsigned int Size , typename Index >
		std::ostream &operator << ( std::ostream &os , MultiIndex< Size , Index > mi )
		{
			os << "{ " << mi[0];
			for( unsigned int i=1 ; i<Size ; i++ ) os << " , " << mi[i];
			return os << " }";
		}

		template< unsigned int Dim , typename Index >
		std::pair< MultiIndex< Dim , Index > , bool > OrientedFaceIndex( SimplexIndex< Dim , Index > si , Index v )
		{
			auto EvenFace = []( SimplexIndex< Dim-1 , Index > face )
				{
					unsigned int count=0;
					while( true )
					{
						unsigned int needsFlip = -1;
						for( unsigned int i=0 ; i<Dim-1 && needsFlip==-1 ; i++ ) if( face[i]>face[i+1] ) needsFlip = i;
						if( needsFlip==-1 ) return (count&1)==0;
						else std::swap( face[needsFlip] , face[needsFlip+1] ) , count++;
					}
				};

			SimplexIndex< Dim-1 , Index > face = OrientedBoundaryFace( si , v );
			return std::make_pair( MultiIndex< Dim , Index >( &face[0] ) , EvenFace(face) );
		}

		template< typename Index >
		std::pair< MultiIndex< 1 , Index > , bool > OrientedFaceIndex( SimplexIndex< 1 , Index > si , Index v )
		{
			if     ( v==0 ) return std::make_pair( MultiIndex< 1 , Index >( &si[1] ) , true  );
			else if( v==1 ) return std::make_pair( MultiIndex< 1 , Index >( &si[0] ) , false );
			else MK_ERROR_OUT( "Bad index: " , v );
			return std::pair< MultiIndex< 1 , Index > , bool >();
		}

		template< unsigned int K , typename Index >
		unsigned int RemoveDuplicates( std::vector< SimplexIndex< K , Index > > &simplexIndices )
		{
			using _Index = MultiIndex< K+1 , Index >;
			using _Map = typename _Index::map;
			_Map _map;

			unsigned int duplicateCount = 0;

			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) _map[ _Index( &simplexIndices[i][0] ) ] = 0;
			Index count = 0;
			for( auto iter=_map.begin() ; iter!=_map.end() ; iter++ ) iter->second = count++;

			std::vector< unsigned int > counts( count , 0 );
			for( unsigned int i=(unsigned int)simplexIndices.size() ; i>0 ; i-- )
			{
				_Index _idx( &simplexIndices[i-1][0] );
				auto iter = _map.find( _idx );
				if( iter==_map.end() ) MK_ERROR_OUT( "Could not find index" );
				if( counts[iter->second] )
				{
					simplexIndices[i-1] = simplexIndices.back();
					simplexIndices.pop_back();
					duplicateCount++;
				}
				else counts[iter->second] = 1;
			}
			return duplicateCount;
		}

		template< unsigned int K , typename Index >
		void Orient( std::vector< SimplexIndex< K , Index > > &simplexIndices )
		{
			using FaceIndex = MultiIndex< K , Index >;
			using FaceMap = typename FaceIndex::map;
			struct FaceData
			{
				Index s1 , s2;					// Indices of the simplices incident on the face
				bool oriented1 , oriented2;		// Flags indicating the orientation of the face within each simplex
				FaceData( void ) : s1(-1) , s2(-1) {}
			};

			FaceMap fMap;

			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) for( Index v=0 ; v<=K ; v++ ) fMap[ OrientedFaceIndex( simplexIndices[i] , v ).first ] = 0;

			Index count = 0;
			for( auto iter=fMap.begin() ; iter!=fMap.end() ; iter++ ) iter->second = count++;

			std::vector< FaceData > fData( count );

			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) for( unsigned int v=0 ; v<=K ; v++ )
			{
				std::pair< FaceIndex , bool > fInfo = OrientedFaceIndex( simplexIndices[i] , v );
				auto iter = fMap.find( fInfo.first );
				if( iter==fMap.end() ) MK_ERROR_OUT( "Could not find edge" );
				if     ( fData[ iter->second ].s1==-1 ) fData[ iter->second ].s1 = i , fData[ iter->second ].oriented1 = fInfo.second;
				else if( fData[ iter->second ].s2==-1 ) fData[ iter->second ].s2 = i , fData[ iter->second ].oriented2 = fInfo.second;
				else MK_ERROR_OUT( "Both faces occupied: " , iter->first , " -> " , simplexIndices[ fData[ iter->second ].s1 ] , " , " , simplexIndices[ fData[ iter->second ].s2 ] , " , " , simplexIndices[i] );
			}


			std::vector< bool > oriented( simplexIndices.size() , false );
			auto NeedsOrienting = [&]( void )
				{
					for( unsigned int i=0 ; i<oriented.size() ; i++ ) if( !oriented[i] ) return (Index)i;
					return (Index)-1;
				};

			Index idx;
			while( (idx=NeedsOrienting())!=-1 )
			{
				std::vector< std::pair< Index , bool > > queue;
				oriented[idx] = true;
				queue.push_back( std::make_pair( idx , true ) );

				while( queue.size() )
				{
					std::pair< Index , bool > s = queue.back();
					queue.pop_back();

					for( unsigned int v=0 ; v<=K ; v++ )
					{
						FaceIndex faceIndex = OrientedFaceIndex( simplexIndices[s.first] , v ).first;
						auto iter = fMap.find( faceIndex );
						if( iter==fMap.end() ) MK_ERROR_OUT( "Could not find face" );
						Index _s = s.first==fData[ iter->second ].s1 ? fData[ iter->second ].s2 : fData[ iter->second ].s1;
						if( _s==-1 || oriented[_s] ) continue;
						oriented[_s] = true;
						if( s.second != ( fData[ iter->second ].oriented1==fData[ iter->second ].oriented2 ) ) queue.push_back( std::make_pair( _s , true ) );
						else
						{
							std::swap( simplexIndices[_s][0] , simplexIndices[_s][1] );
							queue.push_back( std::make_pair( _s , false ) );
						}
					}
				}
			}
		}

		template< unsigned int K , typename Index >
		bool IsOriented( const std::vector< SimplexIndex< K , Index > > &simplexIndices )
		{
			using FaceIndex = MultiIndex< K , Index >;
			using FaceMap = typename FaceIndex::map;
			struct FaceData
			{
				Index s1 , s2;					// Indices of the simplices incident on the face
				bool oriented1 , oriented2;		// Flags indicating the orientation of the face within each simplex
				FaceData( void ) : s1(-1) , s2(-1) {}
			};

			FaceMap fMap;

			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) for( Index v=0 ; v<=K ; v++ ) fMap[ OrientedFaceIndex( simplexIndices[i] , v ).first ] = 0;

			Index count = 0;
			for( auto iter=fMap.begin() ; iter!=fMap.end() ; iter++ ) iter->second = count++;

			std::vector< FaceData > fData( count );

			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) for( unsigned int v=0 ; v<=K ; v++ )
			{
				std::pair< FaceIndex , bool > fInfo = OrientedFaceIndex( simplexIndices[i] , v );
				auto iter = fMap.find( fInfo.first );
				if( iter==fMap.end() ) MK_ERROR_OUT( "Could not find edge" );
				if     ( fData[ iter->second ].s1==-1 ) fData[ iter->second ].s1 = i , fData[ iter->second ].oriented1 = fInfo.second;
				else if( fData[ iter->second ].s2==-1 ) fData[ iter->second ].s2 = i , fData[ iter->second ].oriented2 = fInfo.second;
				else MK_ERROR_OUT
				(
					"Both faces occupied: " ,
					iter->first , " -> " ,
					simplexIndices[ fData[ iter->second ].s1 ] , " , " , simplexIndices[ fData[ iter->second ].s2 ] , " , " , simplexIndices[i]
				);
			}

			for( unsigned int i=0 ; i<fData.size() ; i++ ) if( fData[i].s2!=-1 && fData[i].oriented1==fData[i].oriented2 ) return false;
			return true;
		}

		template< unsigned int K , typename Index >
		std::vector< std::vector< size_t > > FaceAdjacentConnectedComponents( const std::vector< SimplexIndex< K , Index > > &simplexIndices )
		{
			using FaceIndex = MultiIndex< K , Index >;
			using FaceMap = typename FaceIndex::map;

			// Start by computing a mapping from un-orientedfaces to indices
			FaceMap fMap;
			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) for( Index v=0 ; v<=K ; v++ ) fMap[ OrientedFaceIndex( simplexIndices[i] , v ).first ] = 0;
			Index count = 0;
			for( auto iter=fMap.begin() ; iter!=fMap.end() ; iter++ ) iter->second = count++;
			std::vector< std::pair< size_t , size_t > > fData( count , std::pair< size_t , size_t >(-1,-1) );

			// Assign (at most) two simplices to each face
			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) for( Index v=0 ; v<=K ; v++ )
			{
				FaceIndex fi = OrientedFaceIndex( simplexIndices[i] , v ).first;
				typename FaceMap::iterator iter = fMap.find( fi );
				if( iter==fMap.end() ) MK_ERROR_OUT( "Could not find index for face" );
				if     ( fData[ iter->second ].first ==-1 ) fData[ iter->second ].first  = i;
				else if( fData[ iter->second ].second==-1 ) fData[ iter->second ].second = i;
				else MK_ERROR_OUT( "Face occuppied" );
			}


			// The list of (face-adjacent) components
			std::vector< std::vector< size_t > > components;

			// The processed state of the simplex indices (i.e. have they been added to a queue yet)
			std::vector< bool > processedSI( simplexIndices.size() , false );

			// The list of simplices that have not been processed
			std::vector< size_t > unprocessedSI( simplexIndices.size() );
			for( unsigned int i=0 ; i<simplexIndices.size() ; i++ ) unprocessedSI[i] = i;

			// As long at there are indices to process
			while( unprocessedSI.size() )
			{
				// The connected components for this go-around
				std::vector< size_t > component;

				// The queue of indices to propagate to
				std::vector< size_t > Q;

				// Add something into the queue and mark it as processed
				Q.push_back( unprocessedSI.back() );
				processedSI[ unprocessedSI.back() ] = true;
				unprocessedSI.pop_back();
				while( Q.size() )
				{
					// Get the next index of the next simplex in this component and add it to the component
					size_t i = Q.back();
					Q.pop_back();
					component.push_back(i);

					// Add face adjacent neighbors into the queue
					for( Index v=0 ; v<=K ; v++ )
					{
						typename FaceMap::iterator iter = fMap.find( OrientedFaceIndex( simplexIndices[i] , v ).first );
						size_t opposite;
						if( iter==fMap.end() ) MK_ERROR_OUT( "Could not find index for face" );
						if     ( fData[ iter->second ].first ==i ) opposite = fData[ iter->second ].second;
						else if( fData[ iter->second ].second==i ) opposite = fData[ iter->second ].first;
						else MK_ERROR_OUT( "Could not simplex in face" );
						if( opposite!=-1 && !processedSI[opposite] ) Q.push_back( opposite ) , processedSI[opposite] = true;
					}
				}
				// Add the indices of the connected component into the list of connected components
				components.push_back( component );

				// Remove any indices that have already been processed
				for( size_t i=unprocessedSI.size() ; i>0 ; i-- ) if( processedSI[ unprocessedSI[i-1] ] )
				{
					unprocessedSI[i-1] = unprocessedSI.back();
					unprocessedSI.pop_back();
				}
			}

			return components;
		}


		template< unsigned int Dim , unsigned int K > struct HyperCube;

		template< unsigned int Dim > struct HyperCube< Dim , 0 >{};

		template< unsigned int Dim > struct HyperCube< Dim , 1 >
		{
			static const unsigned int K = 1;
			static const unsigned int Size = 1<<K;

			Point< unsigned int , Dim > &operator()( const unsigned int idx[ K ] );
			const Point< unsigned int , Dim > &operator()( const unsigned int idx[ K ] ) const;

			Point< unsigned int , Dim > center( void ) const;

			static void FactorIndex( unsigned int i , unsigned int idx[K] );

			void addSimplices( std::vector< Simplex< unsigned int , Dim , K > > &simplices ) const;

		protected:
			Point< unsigned int , Dim > _p[2];
		};

		template< unsigned int Dim , unsigned int K > struct HyperCube
		{
			static const unsigned int Size = 1<<K;

			Point< unsigned int , Dim > &operator()( const unsigned int idx[ K ] );
			const Point< unsigned int , Dim > &operator()( const unsigned int idx[ K ] ) const;

			Point< unsigned int , Dim > center( void ) const;

			static void FactorIndex( unsigned int i , unsigned int idx[K] );

			void addSimplices( std::vector< Simplex< unsigned int , Dim , K > > &simplices ) const;

		protected:
			HyperCube< Dim , K-1 > _p[2];
		};

		template< typename DataReal , unsigned CoDim , unsigned int Dim , typename Index , typename Data , typename PositionFunction /* = std::function< Point< double , CoDim > ( unsigned int ) > */ > 
		void _AddLevelSet
		(
			std::vector< SimplexIndex< Dim-CoDim , Index > > &levelSetSimplices ,
			SimplexIndex< Dim , Index > si , PositionFunction PF , Point< double , CoDim > isoValue , 
			const std::vector< Data > &inData , typename MultiIndex< CoDim+1 , Index >::map &vMap , std::vector< Data > &outData
		)
		{
			static_assert( CoDim<=Dim , "[ERROR] CoDim cannot exceed Dim" );

			{
				Point< double , CoDim > center;
				for( unsigned int d=0 ; d<=Dim ; d++ ) center += PF( si[d] );
				center /= Dim+1;
				double r2 = 0;
				for( unsigned int d=0 ; d<=Dim ; d++ ) r2 = std::max< double >( r2 , ( center - PF( si[d] ) ).squareNorm() );
				if( ( center - isoValue ).squareNorm()>r2 ) return;
			}

			// If we are at the co-dimension dimension, generate the vertices
			if constexpr( Dim==CoDim )
			{
				auto InList =[&]( Index v )
					{
						bool inList = false;
						for( unsigned int i=0 ; i<levelSetSimplices.size() ; i++ ) inList |= v==levelSetSimplices[i][0];
						return inList;
					};
				MultiIndex< CoDim+1 , Index > idx( &si[0] );
				auto iter = vMap.find( idx );
				// Check if the vertex is already in the map
				if( iter!=vMap.end() && !InList( iter->second ) ) levelSetSimplices.push_back( SimplexIndex< Dim-CoDim , Index >( iter->second ) );
				// Otherwise generate the vertex and add it if it is inside the simplex
				else
				{
					// Construct the CoDim-dimensional simplex
					Simplex< double , CoDim , CoDim > s;
					for( unsigned int i=0 ; i<=CoDim ; i++ ) s[i] = PF( si[i] );

					// Solve for the barycentric coordinates such that:
					//		s[0] + \sum_{i=1}^CoDim a[i] * ( s[i] - s[0] ) = isoValue
					// =>	\sum_{i=1}^CoDim a[i] * ( s[i] - s[0] ) = isoValue - s[0]
					Eigen::Matrix< double , CoDim , CoDim > M;
					for( unsigned int c=0 ; c<CoDim ; c++ ) for( unsigned int r=0 ; r<CoDim ; r++ ) M(r,c) = s[c+1][r] - s[0][r];
					Eigen::Matrix< double , CoDim+1 , 1 > a;
					{
						Eigen::Matrix< double , CoDim , 1 > _a;
						for( unsigned int i=0 ; i<CoDim ; i++ ) _a[i] = isoValue[i] - s[0][i];
						_a = M.fullPivHouseholderQr().solve(_a);
						a[0] = 1.;
						for( unsigned int i=0 ; i<CoDim ; i++ ) a[0] -= _a[i] , a[i+1] = _a[i];
					}

					const double EPS=1e-15;
					bool interior = true;

					bool recheckIndex = false;
					SimplexIndex< Dim , Index > _si = si;
					for( unsigned int i=0 ; i<=CoDim ; i++ )
					{
						interior &= a[i]>=0 && a[i]<=1.;
						if( std::abs( a[i] )<EPS ) a[i] = 0 , _si[i] = -1 , recheckIndex = true;
					}

					if( interior )
					{
						bool done = false;
						if( recheckIndex )
						{
							idx = MultiIndex< CoDim+1 , Index >( &_si[0] );
							auto iter = vMap.find( idx );
							if( iter!=vMap.end() )
							{
								done = true;
								if( !InList( iter->second ) ) levelSetSimplices.push_back( SimplexIndex< Dim-CoDim , Index >( iter->second ) );
							}
						}
						if( !done )
						{
							Index v = (Index)outData.size();
							vMap[ idx ] = v;
							Data d = inData[ si[0] ] * (DataReal)a[0];
							for( unsigned int i=1 ; i<=CoDim ; i++ ) d += inData[ si[i] ] * (DataReal)a[i];
							outData.push_back( d );

							levelSetSimplices.push_back( SimplexIndex< Dim - CoDim , Index >( v ) );
						}
					}
				}
			}
			// In the case of a co-dimension one level-set, we can orient the level-set by keeping the side with values less than the iso-value to one side
			else if constexpr( CoDim==1 && Dim==2 )
			{
				// Massage things in case the level-set passes directly through a vertex
				static const double EPS = 1e-10;
				double cValues[3];
				for( unsigned int i=0 ; i<3 ; i++ )
				{
					cValues[i] = ( PF( si[i] ) - isoValue )[0];
					if( cValues[i]==0 )
					{
						MK_WARN_ONCE( "Offsetting zero value: " , cValues[i] );
						cValues[i] = EPS;
					}
				}
				// Find the index of the singleton vertex
				unsigned int ii = -1;
				for( unsigned int d=0 ; d<3 ; d++ ) if( cValues[d]*cValues[(d+1)%3]<0 && cValues[d]*cValues[(d+2)%3]<0 ){ ii = d ; break; }
				if( ii==-1 ) return;

				SimplexIndex< 1 , Index > edge , isoEdge;
				edge[0] = si[ii];
				for( unsigned int i=0 ; i<2 ; i++ )
				{
					edge[1] = si[ (ii+1+i)%3 ];
					MultiIndex< 2 , Index > idx( &edge[0] );
					auto iter = vMap.find( idx );
					if( iter!=vMap.end() ) isoEdge[i] = iter->second;
					else
					{
						// Solve for s s.t.
						//	(1-s) * cValues[ii] + s * cValues[(ii+1+i)%3 ] = 0;
						//	s * ( cValues[(ii+1+i)%3 ] - cValues[ii] ) = - cValues[ii]
						//	s = - cValues[ii] / ( cValues[(ii+1+i)%3 ] - cValues[ii] )
						double s = - cValues[ii] / ( cValues[(ii+1+i)%3] - cValues[ii] );
						Index v = (Index)outData.size();
						isoEdge[i] = vMap[ idx ] = v;
						Data d = inData[ edge[0] ] * (DataReal)( 1. - s ) + inData[ edge[1] ] * (DataReal)s;
						outData.push_back( d );
					}
				}
				if( cValues[ii]<0 ) std::swap( isoEdge[0] , isoEdge[1] );
				levelSetSimplices.push_back( isoEdge );
			}
			else
			{
				// Iterate over all the faces and accumulate the level-sets
				std::vector< SimplexIndex< Dim - CoDim-1 , Index > > _levelSetSimplices;
				for( unsigned int i=0 ; i<=Dim ; i++ ) _AddLevelSet< DataReal , CoDim , Dim-1 , Index >( _levelSetSimplices , OrientedBoundaryFace( si , i ) , PF , isoValue , inData , vMap , outData );
				if( _levelSetSimplices.size() )
				{
					unsigned int v = _levelSetSimplices[0][0];
					for( unsigned int i=1 ; i<_levelSetSimplices.size() ; i++ )
					{
						bool containsV = false;
						for( unsigned int j=0 ; j<(Dim-CoDim) ; j++ ) containsV |= _levelSetSimplices[i][j]==v;
						if( !containsV )
						{
							SimplexIndex< Dim-CoDim , Index > _si;
							_si[0] = v;
							for( unsigned int j=0 ; j<(Dim-CoDim) ; j++ ) _si[j+1] = _levelSetSimplices[i][j];
							levelSetSimplices.push_back( _si );
						}
					}
				}
			}
		}

		template< typename DataReal , unsigned CoDim , unsigned int Dim , typename Index , typename Data , typename PositionFunctor /* = std::function< Point< double , CoDim > ( unsigned int ) > */ > 
		SimplicialMesh< Dim - CoDim , Index , Data > LevelSet( const SimplicialMesh< Dim , Index , Data > &sMesh , PositionFunctor PF , Point< double , CoDim > isoValue )
		{
			static_assert( CoDim>0 , "[ERROR] CoDim should be positive" );
			static_assert( CoDim<=Dim , "[ERROR] CoDim cannot exceed Dim" );
			SimplicialMesh< Dim - CoDim , Index , Data > levelSet;
			typename MultiIndex< CoDim+1 , Index >::map vMap;
			for( unsigned int i=0 ; i<sMesh.simplexIndices.size() ; i++ ) _AddLevelSet< DataReal , CoDim , Dim >( levelSet.simplexIndices , sMesh.simplexIndices[i] , PF , isoValue , sMesh.vertices , vMap , levelSet.vertices );
			return levelSet;
		}

		//////////////////////////
		// HyperCube< Dim , 1 > //
		//////////////////////////
		template< unsigned int Dim >
		Point< unsigned int , Dim > &HyperCube< Dim , 1 >::operator()( const unsigned int idx[ K ] ){ return _p[ idx[0] ]; }

		template< unsigned int Dim >
		const Point< unsigned int , Dim > &HyperCube< Dim , 1 >::operator()( const unsigned int idx[ K ] ) const { return _p[ idx[0] ]; }

		template< unsigned int Dim >
		Point< unsigned int , Dim > HyperCube< Dim , 1 >::center( void ) const
		{
			Point< unsigned int , Dim > c = _p[0] + _p[1];
			for( unsigned int d=0 ; d<Dim ; d++ ) c[d] /= 2;
			return c;
		}

		template< unsigned int Dim >
		void HyperCube< Dim , 1 >::FactorIndex( unsigned int i , unsigned int idx[ K ] ){ idx[0] = i; }

		template< unsigned int Dim >
		void HyperCube< Dim , 1 >::addSimplices( std::vector< Simplex< unsigned int , Dim , K > > &simplices ) const
		{
			Simplex< unsigned int , Dim , K > s;
			s[0] = _p[0] , s[1] = _p[1];
			simplices.push_back( s );
		}

		template< unsigned int Dim >
		SimplicialMesh< Dim , unsigned int , Point< double , Dim > > RegularGridTriangulation( unsigned int res , bool orient , bool addCenters )
		{
			auto FactorIndex = [res]( size_t e )
				{
					Point< unsigned int , Dim > idx;
					for( unsigned int d=0 ; d<Dim ; d++ ){ idx[d] = e%res ; e /= res; }
					return idx;
				};

			size_t eNum = 1;
			for( unsigned int d=0 ; d<Dim ; d++ ) eNum *= res;

			return RegularSubGridTriangulation< Dim >( eNum , FactorIndex , orient , addCenters );
		}

		template< unsigned int Dim , typename IndexFunctor /* = std::function< Point< unsigned int , Dim > (size_t) > */>
		SimplicialMesh< Dim , unsigned int , Point< double , Dim > > RegularSubGridTriangulation( size_t eNum , IndexFunctor && F , bool orient , bool addCenters )
		{
			static_assert( std::is_convertible_v< IndexFunctor , std::function< Point< unsigned int , Dim > ( size_t ) > > , "[ERROR] IndexFunctor poorly formed" );
			std::vector< Simplex< unsigned int , Dim , Dim > > simplices;

			// Create the simplicial decomposition of the canonical hyper-cube
			if( addCenters )
			{
				HyperCube< Dim , Dim > cube;
				unsigned int idx[Dim];
				for( unsigned int i=0 ; i<(1<<Dim) ; i++ )
				{
					HyperCube< Dim , Dim >::FactorIndex( i , idx );
					Point< unsigned int , Dim > p;
					for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = idx[d] * 2;
					cube( idx ) = p;
				}
				cube.addSimplices( simplices );
			}
			else
			{
				HyperCubeSimplices< Dim > hcSimplices;
				simplices.resize( HyperCubeSimplices< Dim >::SimplexNum );
				for( unsigned int i=0 ; i<HyperCubeSimplices< Dim >::SimplexNum ; i++ ) simplices[i] = hcSimplices.simplex(i);
			}

			if( orient )
				for( unsigned int i=0 ; i<simplices.size() ; i++ )
				{
					SquareMatrix< double , Dim > M;
					for( unsigned int j=0 ; j<Dim ; j++ ) for( unsigned int k=0 ; k<Dim ; k++ ) M(j,k) = (double)simplices[i][j+1][k] - (double)simplices[i][0][k];
					if( M.determinant()<0 ) std::swap( simplices[i][0] , simplices[i][1] );
				}

			struct PointHasher
			{
				size_t operator()( const Point< unsigned int , Dim > &idx ) const
				{
					// from boost::hash_combine
					size_t hash = 0;
					for( unsigned int i=0 ; i<Dim ; i++ ) hash ^= std::hash< unsigned int >()( idx[i] ) + 0x9e3779b9 + (hash<<6) + (hash>>2);
					return hash;
				}
			};

			struct PointEquality
			{
				bool operator()( Point< unsigned int , Dim > p1 , Point< unsigned int , Dim > p2 ) const
				{
					for( unsigned int d=0 ; d<Dim ; d++ ) if( p1[d]!=p2[d] ) return false;
					return true;
				}
			};
			std::unordered_map< Point< unsigned int , Dim > , unsigned int , PointHasher , PointEquality > vMap;

			SimplicialMesh< Dim , unsigned int , Point< double , Dim > > mesh;

			// Iterate over each cell
			for( size_t e=0 ; e<eNum ; e++ )
			{
				Point< unsigned int , Dim > idx = F( e );
				for( unsigned int i=0 ; i<simplices.size() ; i++ )
				{
					SimplexIndex< Dim , unsigned int > si;
					for( unsigned int j=0 ; j<=Dim ; j++ )
					{
						Point< unsigned int , Dim > v = simplices[i][j];
						if( addCenters ) for( unsigned int d=0 ; d<Dim ; d++ ) v[d] += idx[d]*2;
						else             for( unsigned int d=0 ; d<Dim ; d++ ) v[d] += idx[d];
						auto iter = vMap.find( v );
						if( iter==vMap.end() )
						{
							vMap[v] = si[j] = (unsigned int)mesh.vertices.size();
							if( addCenters ) mesh.vertices.push_back( Point< double , Dim >( v ) / 2. );
							else             mesh.vertices.push_back( Point< double , Dim >( v ) );
						}
						else si[j] = iter->second;
					}
					mesh.simplexIndices.push_back( si );
				}
			}

			return mesh;
		}

		//////////////////////////
		// HyperCube< Dim , K > //
		//////////////////////////
		template< unsigned int Dim , unsigned int K >
		Point< unsigned int , Dim > &HyperCube< Dim , K >::operator()( const unsigned int idx[ K ] ){ return _p[ idx[0] ]( idx+1 ); }

		template< unsigned int Dim , unsigned int K >
		const Point< unsigned int , Dim > &HyperCube< Dim , K >::operator()( const unsigned int idx[ K ] ) const { return _p[ idx[0] ]( idx+1 ); }

		template< unsigned int Dim , unsigned int K >
		Point< unsigned int , Dim > HyperCube< Dim , K >::center( void ) const
		{
			Point< unsigned int , Dim > c = _p[0].center() + _p[1].center();
			for( unsigned int d=0 ; d<Dim ; d++ ) c[d] /= 2;
			return c;
		}

		template< unsigned int Dim , unsigned int K >
		void HyperCube< Dim , K >::FactorIndex( unsigned int i , unsigned int idx[K] ){ idx[0] = i&1 ; HyperCube< Dim , K-1 >::FactorIndex( i>>1 , idx+1 ); }

		template< unsigned int Dim , unsigned int K >
		void HyperCube< Dim , K >::addSimplices( std::vector< Simplex< unsigned int , Dim , K > > &simplices ) const
		{
			// The center of the cube
			Point< unsigned int , Dim > c = center();
			// The simplices of the boundary
			std::vector< Simplex< unsigned int , Dim , K-1 > > _simplices;

			// Add the side faces
			for( unsigned int k=0 ; k<K ; k++ )
			{
				unsigned int idx0[Dim] , idx1[Dim];
				unsigned int _idx[Dim-1];
				idx0[k] = 0 , idx1[k] = 1;
				HyperCube< Dim , K-1 > cube0 , cube1;
				for( unsigned int i=0 ; i<( 1<<(K-1) ) ; i++ )
				{
					HyperCube< Dim , K-1 >::FactorIndex( i , _idx );
					for( unsigned int _k=0 , __k=0 ; _k<K ; _k++ ) if( _k!=k ) idx0[_k] = idx1[_k] = _idx[__k++];
					cube0( _idx ) = this->operator ()( idx0 );
					cube1( _idx ) = this->operator ()( idx1 );
				}
				cube0.addSimplices( _simplices );
				cube1.addSimplices( _simplices );
			}

			size_t sz = simplices.size();
			simplices.resize( sz + _simplices.size() );
			for( unsigned int i=0 ; i<_simplices.size() ; i++ )
			{
				simplices[ sz + i ][0] = c;
				for( unsigned int k=0 ; k<K ; k++ ) simplices[sz+i][k+1] = _simplices[i][k];
			}
		}

		////////////////
		// MultiIndex //
		////////////////
		template< unsigned int Size , typename Index >
		template< typename ... UInts >
		MultiIndex< Size , Index >::MultiIndex( UInts ... indices )
		{
			static_assert( sizeof ... ( UInts )==Size , "[ERROR] Wrong number of indices" );
			const Index _indices[] = { (Index)indices... };
			_init( _indices );
		}
		template< unsigned int Size , typename Index >
		bool MultiIndex< Size , Index >::operator < ( const MultiIndex &idx ) const
		{
			for( unsigned int i=0 ; i<Size ; i++ )
			{
				if( _indices[i]<idx._indices[i] ) return true;
				else if( _indices[i]>idx._indices[i] ) return false;
			}
			return false;
		};

		template< unsigned int Size , typename Index >
		bool MultiIndex< Size , Index >::operator == ( const MultiIndex &idx ) const
		{
			return !( (*this)<idx ) && !( idx<(*this) );
		}

		template< unsigned int Size , typename Index >
		void MultiIndex< Size , Index >::_init( const Index indices[] )
		{
			memcpy( _indices , indices, sizeof(Index) * Size );
			std::sort( _indices , _indices + Size , []( unsigned int v1 , unsigned int v2 ){ return v1<v2; } );
		}
	}
}
#endif // MARCHING_SIMPLICES_INCLUDED
