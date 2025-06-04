/*
Copyright (c) 2022, Michael Kazhdan
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

#ifndef HYPER_CUBE_SIMPLICES_INCLUDED
#define HYPER_CUBE_SIMPLICES_INCLUDED

#include <unordered_map>
#include "Misha/Geometry.h"
#include "Misha/RegularGrid.h"

namespace MishaK
{
	template< unsigned int Dim >
	struct HyperCube
	{
		template< unsigned int K >
		static constexpr unsigned int SubCellCount( void )
		{
			if constexpr( K==Dim ) return 1;
			else if constexpr( K==0 ) return 1<<Dim;
			else return HyperCube< Dim-1 >::template SubCellCount< K >() * 2 + HyperCube< Dim-1 >::template SubCellCount< K-1 >();
		}
	};

	template< unsigned int _Dim >
	struct HyperCubeSimplices
	{
		static const unsigned int Dim = _Dim;

		static const unsigned int SimplexNum = HyperCubeSimplices< Dim-1 >::SimplexNum * Dim;
		SimplexIndex< Dim > &operator[]( unsigned int idx ){ return _simplexIndices[idx]; }
		const SimplexIndex< Dim > &operator[]( unsigned int idx ) const { return _simplexIndices[idx]; }

		static const unsigned int BoundarySimplexNum = SimplexNum * 2;
		SimplexIndex< Dim-1 > &boundarySimplexIndex( unsigned int idx ){ return _boundarySimplexIndices[idx]; }
		const SimplexIndex< Dim-1 > &boundarySimplexIndex( unsigned int idx ) const { return _boundarySimplexIndices[idx]; }

		static const unsigned int VertexNum = HyperCube< Dim >::template SubCellCount< 0 >();
		Point< unsigned int , Dim > vertex( unsigned int v ) const { return _vertices[v]; }

		Simplex< unsigned int , Dim , Dim > simplex( unsigned int i ) const
		{
			Simplex< unsigned int , Dim , Dim > s;
			for( unsigned int d=0 ; d<=Dim ; d++ ) s[d] = _vertices[ _simplexIndices[i][d] ];
			return s;
		}

		Simplex< unsigned int , Dim , Dim-1 > boundarySimplex( unsigned int i ) const
		{
			Simplex< unsigned int , Dim , Dim-1 > s;
			for( unsigned int d=0 ; d<Dim ; d++ ) s[d] = _vertices[ _boundarySimplexIndices[i][d] ];
			return s;
		}

		HyperCubeSimplices( void )
		{
			if constexpr( Dim==1 ) // Base case
			{
				_vertices[0][0] = 0;
				_vertices[1][0] = 1;

				_simplexIndices[0][0] = 0;
				_simplexIndices[0][1] = 1;

				_boundarySimplexIndices[0][0] = 0;
				_boundarySimplexIndices[1][0] = 1;
			}
			else // Inductive case
			{
				HyperCubeSimplices< Dim-1 > hcs;

				// Computes the index associated to a vertex (on the unit cube)
				auto VertexIndex = [&]( Point< unsigned int , Dim > p ) -> unsigned int
					{
						for( unsigned int v=0 ; v<VertexNum ; v++ )
						{
							bool isEqual = true;
							for( unsigned int d=0 ; d<Dim ; d++ ) isEqual &= p[d]==_vertices[v][d];
							if( isEqual ) return v;
						}
						MK_ERROR_OUT( "Could not match vertex: " , p );
						return -1;
					};

				// Maps a (Dim-1)-dimensional point to a Dim-dimensional point with a value of "off" in the dir-th component
				auto PromoteVertex = [&]( unsigned int dir , unsigned int off , Point< unsigned int , Dim-1 > _p )
					{
						Point< unsigned int , Dim > p;
						for( unsigned int d=0 , idx=0 ; d<Dim ; d++ )
						{
							if( d==dir ) p[d] = off;
							else p[d] = _p[idx++];
						}
						return p;
					};

				// Maps a (Dim-1)-dimensional simplex in (Dim-1)-dimensional space to a Dim-dimensional simplex in Dim-dimensional space by:
				// 1. copying (and promoting) the vertices of the (Dim-1)-dimensional simplex
				// 2. adding a vertex at the origin
				auto PromoteSimplex = [&]( unsigned int dir , Simplex< unsigned int , Dim-1 , Dim-1 > _s )
					{
						Simplex< unsigned int , Dim , Dim > s;
						for( unsigned int d=0 ; d<Dim ; d++ ) s[d] = PromoteVertex( dir , 1 , _s[d] );
						s[Dim] = Point< unsigned int , Dim >();
						return s;
					};

				// Maps a (Dim-1)-dimensional simplex in (Dim-1)-dimensional space to a (Dim-1)-dimensional simplex in Dim-dimensional space by promiting the vertices
				auto PromoteBoundarySimplex = [&]( unsigned int dir , unsigned int off , Simplex< unsigned int , Dim-1 , Dim-1 > _s )
					{
						Simplex< unsigned int , Dim , Dim-1 > s;
						for( unsigned int d=0 ; d<Dim ; d++ ) s[d] = PromoteVertex( dir , off , _s[d] );
						return s;
					};

				// Set the vertices
				for( unsigned int v=0 ; v<HyperCubeSimplices< Dim-1 >::VertexNum ; v++ )
				{
					Point< unsigned int , Dim-1 > p = hcs.vertex( v );
					for( unsigned int d=0 ; d<Dim-1 ; d++ ) _vertices[v][d] = _vertices[v+HyperCubeSimplices< Dim-1 >::VertexNum][d] = p[d];
					_vertices[v+0*HyperCubeSimplices< Dim-1 >::VertexNum][Dim-1] = 0;
					_vertices[v+1*HyperCubeSimplices< Dim-1 >::VertexNum][Dim-1] = 1;
				}

				// Set the simplices
				for( unsigned int dir=0 , idx=0 ; dir<Dim ; dir++ ) for( unsigned int i=0 ; i<HyperCubeSimplices< Dim-1 >::SimplexNum ; i++ , idx++ )
				{
					SimplexIndex< Dim-1 > _si = hcs[i];
					Simplex< unsigned int , Dim-1 , Dim-1 > _s;
					for( unsigned int d=0 ; d<Dim ; d++ ) _s[d] = hcs.vertex( _si[d] );
					Simplex< unsigned int , Dim , Dim > s = PromoteSimplex( dir , _s );
					for( unsigned int d=0 ; d<=Dim ; d++ ) _simplexIndices[idx][d] = VertexIndex( s[d] );
				}

				// Set (and orient) the boundary simplices
				Point< double, Dim > center;
				for( unsigned int d=0 ; d<Dim ; d++ ) center[d] = 0.5;
				for( unsigned int dir=0 , idx=0 ; dir<Dim ; dir++ ) for( unsigned int off=0 ; off<2 ; off++ ) for( unsigned int i=0 ; i<HyperCubeSimplices< Dim-1 >::SimplexNum ; i++ , idx++ )
				{
					SimplexIndex< Dim-1 > _si = hcs[i];
					Simplex< unsigned int , Dim-1 , Dim-1 > _s;
					for( unsigned int d=0 ; d<Dim ; d++ ) _s[d] = hcs.vertex( _si[d] );
					Simplex< unsigned int , Dim , Dim-1 > s = PromoteBoundarySimplex( dir , off , _s );
					for( unsigned int d=0 ; d<Dim ; d++ ) _boundarySimplexIndices[idx][d] = VertexIndex( s[d] );
					{
						Simplex< double , Dim , Dim-1 > temp;
						for( unsigned int d=0 ; d<Dim ; d++ ) temp[d] = Point< double , Dim >( s[d] );
						Point< double , Dim > c = temp.center();
						Point< double , Dim > n = temp.normal();
						if( Point< double , Dim >::Dot( c-center , n )<0 ) std::swap( _boundarySimplexIndices[idx][0] , _boundarySimplexIndices[idx][1] );
					}
				}
			}
		}

	protected:
		SimplexIndex< Dim > _simplexIndices[ SimplexNum ];
		SimplexIndex< Dim-1 > _boundarySimplexIndices[ BoundarySimplexNum ];
		Point< unsigned int , Dim > _vertices[ VertexNum ];
	};
	template<>
	struct HyperCubeSimplices< 0 >{ static const unsigned int SimplexNum = 1; };


	template< unsigned int Dim >
	struct CellSimplices
	{
		static const unsigned int SimplexNum = HyperCubeSimplices< Dim >::SimplexNum;
		static const unsigned int VertexNum = HyperCubeSimplices< Dim >::VertexNum;
		static const HyperCubeSimplices< Dim > hcs;

		SimplexIndex< Dim > si[ SimplexNum ];
		Simplex< double , Dim , Dim > s[ SimplexNum ];
		CellSimplices( typename RegularGrid< Dim >::Range range )
		{
			for( unsigned int d=0 ; d<Dim ; d++ ) _off[d] = range.first[d] , _res[d] = range.second[d]-range.first[d];
		}

		unsigned int index( Point< unsigned int , Dim > I )
		{
			unsigned int i=0;
			for( int d=(int)Dim-1 ; d>=0 ; d-- ) i = i * (_res[d]) + ( I[d] - _off[d] );
			return i;
		}

		void set( Point< unsigned int , Dim > I )
		{
			unsigned int indices[ VertexNum ];
			Point< double , Dim > positions[ VertexNum ];

			for( unsigned int n=0 ; n<VertexNum ; n++ )
			{
				Point< unsigned int , Dim > v = hcs.vertex( n );
				indices[n] = index( I + v );
				positions[n] = Point< double , Dim >( I + v );
			}


			for( unsigned int n=0 ; n<SimplexNum ; n++ )
			{
				SimplexIndex< Dim > _si = hcs[n];
				for( unsigned int d=0 ; d<=Dim ; d++ )  si[n][d] = indices[ _si[d] ] , s[n][d] = positions[ _si[d] ];
			}
		}

		Point< double , Dim+1 > simplexValues( const double *cellValues , unsigned int n ) const
		{
			SimplexIndex< Dim > _si = hcs[n];
			Point< double , Dim+1 > values;
			for( unsigned int d=0 ; d<=Dim ; d++ ) values[d] = cellValues[ _si[d] ];
			return values;
		}
	protected:
		unsigned int _res[Dim] , _off[Dim];
	};
	template< unsigned int Dim >
	const HyperCubeSimplices< Dim > CellSimplices< Dim >::hcs;

	template< unsigned int Dim >
	struct CellFaceSimplices
	{
		static const unsigned int SimplexNum = HyperCubeSimplices< Dim >::BoundarySimplexNum;
		static const unsigned int VertexNum = HyperCubeSimplices< Dim >::VertexNum;
		static const HyperCubeSimplices< Dim > hcs;

		SimplexIndex< Dim-1 > si[ SimplexNum ];
		Simplex< double , Dim , Dim-1 > s[ SimplexNum ];

		CellFaceSimplices( typename RegularGrid< Dim >::Range range )
		{
			for( unsigned int d=0 ; d<Dim ; d++ ) _off[d] = range.first[d] , _res[d] = range.second[d]-range.first[d];
		}

		unsigned int index( Point< unsigned int , Dim > I )
		{
			unsigned int i=0;
			for( int d=(int)Dim-1 ; d>=0 ; d-- ) i = i * (_res[d]) + ( I[d] - _off[d] );
			return i;
		}

		void set( Point< unsigned int , Dim > I )
		{
			unsigned int indices[ VertexNum ];
			Point< double , Dim > positions[ VertexNum ];

			for( unsigned int n=0 ; n<VertexNum ; n++ )
			{
				Point< unsigned int , Dim > v = hcs.vertex( n );
				indices[n] = index( I + v );
				positions[n] = Point< double , Dim >( I + v );
			}

			for( unsigned int n=0 ; n<SimplexNum ; n++ )
			{
				SimplexIndex< Dim-1 > _si = hcs.boundarySimplexIndex(n);
				for( unsigned int d=0 ; d<Dim ; d++ )  si[n][d] = indices[ _si[d] ] , s[n][d] = positions[ _si[d] ];
			}
		}

		Point< double , Dim > simplexValues( const double *cellValues , unsigned int n ) const
		{
			SimplexIndex< Dim-1 > _si = hcs.boundarySimplexIndex( n );
			Point< double , Dim > values;
			for( unsigned int d=0 ; d<Dim ; d++ ) values[d] = cellValues[ _si[d] ];
			return values;
		}
	protected:
		unsigned int _res[Dim] , _off[Dim];
	};
	template< unsigned int Dim >
	const HyperCubeSimplices< Dim > CellFaceSimplices< Dim >::hcs;
}
#endif // HYPER_CUBE_SIMPLICES_INCLUDED