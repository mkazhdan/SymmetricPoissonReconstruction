#ifndef UNSIGNED_NORMALS_INCLUDED

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include "Misha/Miscellany.h"
#include "Misha/Geometry.h"
#include "KDTree.h"

namespace MishaK
{
	namespace UnsignedNormals
	{
		template< typename Real , unsigned Dim >
		std::vector< SquareMatrix< Real , Dim > > GetNormalCovariances( const std::vector< Point< Real , Dim > > &points , unsigned int nnNum , bool verbose );

		template< typename Real , unsigned int Dim >
		struct Implicit
		{
			virtual Real operator() ( Point< Real , Dim > p ) const = 0;
			virtual Point< Real , Dim > gradient( Point< Real , Dim > p ) const = 0;
		};

		// Represents the implicit function F(p) = < l , p > + c;
		template< typename Real , unsigned int Dim >
		struct LinearFit : public Implicit< Real , Dim > , VectorSpace< Real , LinearFit< Real , Dim > >
		{
			LinearFit( Point< Real , Dim > p , Point< Real , Dim > n );
			LinearFit( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs );
			Real operator() ( Point< Real , Dim > p ) const { return Point< Real , Dim >::Dot( p , _l ) + _c; }
			Point< Real , Dim > gradient( Point< Real , Dim > p ) const { return _l; }
			void normalize( void ){ _l /= (Real)sqrt( Point< Real , Dim >::SquareNorm( _l ) ); }

			Real &c( void ){ return _c; }
			const Real &c( void ) const { return _c; }
			Point< Real , Dim > &l( void ){ return _l; }
			const Point< Real , Dim > &l( void ) const { return _l; }

			void Add( const LinearFit &f );
			void Scale( Real s );

		protected:
			Real _c;
			Point< Real , Dim > _l;
			void _init( Point< Real , Dim > p );
		};

		// Represents the implicit function F(p) = < p , Q * p > + < g , p > - c;
		template< typename Real , unsigned int Dim >
		struct QuadraticFit : public LinearFit< Real , Dim > , public VectorSpace< Real , QuadraticFit< Real , Dim > >
		{
			QuadraticFit( Point< Real , Dim > p , Point< Real , Dim > n , const std::vector< Point< Real , Dim > > &nbrs , Real regularizationWeight=0 );
			QuadraticFit( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs , Real regularizationWeight=0 );
			Real operator() ( Point< Real , Dim > p ) const { return Point< Real , Dim >::Dot(p,_q*p) + Point< Real , Dim >::Dot( p , _l ) + _c; }
			Point< Real , Dim > gradient( Point< Real , Dim > p ) const { return 2*_q*p + _l; }

			void Add( const QuadraticFit &f );
			void Scale( Real s );

			SquareMatrix< Real , Dim > &q( void ){ return _q; }
			const SquareMatrix< Real , Dim > &q( void ) const { return _q; }
		protected:
			QuadraticFit( void ){}

			using LinearFit< Real , Dim >::_c;
			using LinearFit< Real , Dim >::_l;
			SquareMatrix< Real , Dim > _q;
			void _init( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs , Real regularizationWeight );
		};

		// Represents the implicit function F(p) = < p , Q * p > + < g , p > - c;
		template< typename Real , unsigned int Dim >
		struct QuadraticGraphFit : public LinearFit< Real , Dim > , public VectorSpace< Real , QuadraticGraphFit< Real , Dim > >
		{
			QuadraticGraphFit( Point< Real , Dim > p , Point< Real , Dim > n , const std::vector< Point< Real , Dim > > &nbrs );
			QuadraticGraphFit( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs );
			Real operator() ( Point< Real , Dim > p ) const { return Point< Real , Dim >::Dot(p,_q*p) + Point< Real , Dim >::Dot( p , _l ) + _c; }
			Point< Real , Dim > gradient( Point< Real , Dim > p ) const { return 2*_q*p + _l; }

			void Add( const QuadraticGraphFit &f );
			void Scale( Real s );

			SquareMatrix< Real , Dim > &q( void ){ return _q; }
			const SquareMatrix< Real , Dim > &q( void ) const { return _q; }
		protected:
			using LinearFit< Real , Dim >::_c;
			using LinearFit< Real , Dim >::_l;
			SquareMatrix< Real , Dim > _q;

			QuadraticGraphFit();

			void _init( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs );
		};

		////////////////////
		// Implementation //
		////////////////////

		///////////////
		// LinearFit //
		///////////////
		template< typename Real , unsigned int Dim >
		LinearFit< Real , Dim >::LinearFit( Point< Real , Dim > p , Point< Real , Dim > n )
		{
			_l = n / (Real)sqrt( Point< Real , Dim >::SquareNorm( n ) );
			_init( p );
		}
		template< typename Real , unsigned int Dim >
		LinearFit< Real , Dim >::LinearFit( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs )
		{
			SquareMatrix< Real , Dim > _C;
			for( unsigned int i=0 ; i<nbrs.size() ; i++ ) _C += OuterProduct( nbrs[i]-p , nbrs[i]-p );

			Eigen::Matrix< Real , Dim , Dim > C;
			for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) C(i,j) = _C(j,i);
			Eigen::SelfAdjointEigenSolver< Eigen::Matrix< Real , Dim , Dim > > solver( C );
			auto eVector = solver.eigenvectors().col(0);
			for( unsigned int d=0 ; d<Dim ; d++ ) _l[d] = eVector[d];
			_init( p );
		}

		template< typename Real , unsigned int Dim >
		void LinearFit< Real , Dim >::_init( Point< Real , Dim > p ){ _c = -Point< Real , Dim >::Dot( p , _l ); }

		template< typename Real , unsigned int Dim >
		void LinearFit< Real , Dim >::Add( const LinearFit &lf ){ _l += lf._l ,  _c += lf._c; }

		template< typename Real , unsigned int Dim >
		void LinearFit< Real , Dim >::Scale( Real s ){ _l *=s , _c *=s; }


		//////////////////
		// QuadraticFit //
		//////////////////
		template< typename Real , unsigned int Dim >
		QuadraticFit< Real , Dim >::QuadraticFit( Point< Real , Dim > p , Point< Real , Dim > n , const std::vector< Point< Real , Dim > > &nbrs , Real regularizationWeight )
			: LinearFit< Real , Dim >( p , n ) { _init( p , nbrs , regularizationWeight ); }

		template< typename Real , unsigned int Dim >
		QuadraticFit< Real , Dim >::QuadraticFit( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs , Real regularizationWeight )
			: LinearFit< Real , Dim >( p , nbrs ) { _init( p , nbrs , regularizationWeight ); }

		template< typename Real , unsigned int Dim >
		void QuadraticFit< Real , Dim >::_init( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs , Real regularizationWeight )
		{
			static const unsigned int OuterProductDim = ( Dim * ( Dim+1 ) ) / 2;
			SquareMatrix< Real , OuterProductDim > M;
			for( unsigned int i=0 , idx=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<=i ; j++ , idx++ ) M(idx,idx) = i==j ? 1 : 2;

			auto _OuterProduct = [&]( Point< Real , Dim > q ) -> Point< Real , OuterProductDim >
				{
					Point< Real , OuterProductDim > op;
					for( unsigned int i=0 , idx=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<=i ; j++ , idx++ ) op[idx] = q[i] * q[j];
					return op;
				};

			auto _SymmetricMatrix = [&]( Point< Real , OuterProductDim > q ) -> SquareMatrix< Real , Dim >
				{
					SquareMatrix< Real , Dim > A;
					for( unsigned int i=0 , idx=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<=i ; j++ , idx++ ) A(i,j) = A(j,i) = q[idx];
					return A;
				};

			// Denote by {p} the outer product p \otimes p
			// Denote by [A] the representation of a symmetric matrix A as a vetor
			// Denote by d_i = p_i - p
			// 
			// E(Q) = \sum_i [ < d_i , Q * d_i > + < L , d_i > + C ]^2
			//      = \sum_i [ Tr( Q * {d_i} ) + d_i^t * L + C ]^2
			//      = \sum_i [ [Q]^t * M * [{d_i}] + (d_i)^t * L + C ]^2
			//      = ( \sum_i [Q]^t * M * [{d_i}] * [{d_i}]^t * M * [Q] + 2 * [Q]^t * M * [{d_i}] * ( d_i^t * L + C ) + ... )
			// Taking the derivative w.r.t. [Q] and setting to zero gives:
			//    0 = \sum_i 2 * M * [{d_i}] * [{d_i}]^t * M * [Q] + 2 * M * [{d_i}] * ( d_i^t * L + C )
			//  [Q] = - ( \sum_i 2 * M * [{d_i}] * [{d_i}]^t * M )^{-1} * ( \sum_i 2 * M * [{d_i}] * ( d_i^t * L + C ) )
			//      = - ( \sum_i [{d_i}] * [{d_i}]^t * M )^{-1} * ( \sum_i [{d_i}] * ( d_i^t * L + C ) )
			// Adding a regularization weight:
			// E(Q) = ... + rWeight [Q]^t * M * [Q]
			// Taking the derivative w.r.t. [Q] and setting to zero gives
			//    0 = \sum_i 2 * M * [{d_i}] * [{d_i}]^t * M * [Q] + 2 * M * [{d_i}] * ( d_i^t * L + C ) + 2 * rWeight * M * [Q]
			//  [Q] = - ( \sum_i 2 * M * [{d_i}] * [{d_i}]^t * M + 2 * rWeight * M )^{-1} * ( \sum_i 2 * M * [{d_i}] * ( d_i^t * L + C ) )
			//      = - ( \sum_i [{d_i}] * [{d_i}]^t * M + rWeight )^{-1} * ( \sum_i [{d_i}] * ( d_i^t * L + C ) )

			SquareMatrix< Real , OuterProductDim > A;
			Point< Real , OuterProductDim > b;
			for( unsigned int i=0 ; i<nbrs.size() ; i++ )
			{
				Point< Real , Dim > d = nbrs[i] - p;
				Point< Real , OuterProductDim > _d = _OuterProduct(d);
				A += OuterProduct(_d,_d) * M  , b += _d * ( Point< Real , Dim >::Dot( d , _l ) + _c );
			}
			A /= (Real)nbrs.size() , b /= (Real)nbrs.size();
			A += SquareMatrix< Real , OuterProductDim >::Identity() * regularizationWeight;
			_q = _SymmetricMatrix( - A.inverse() * b );

			// Adjust the linear and constant terms
			// (q-p)^t * Q * (q-p) = q^t * Q * q - 2 q^t * Q * p + p^t * Q * p
			_l -= 2 * _q * p;
			_c += Point< Real , Dim >::Dot( p , _q * p );
		}

		template< typename Real , unsigned int Dim >
		void QuadraticFit< Real , Dim >::Add( const QuadraticFit &f ){ _q += f._q , _l += f._l ,  _c += f._c; }

		template< typename Real , unsigned int Dim >
		void QuadraticFit< Real , Dim >::Scale( Real s ){ _q *=s , _l *=s , _c *=s; }

		///////////////////////
		// QuadraticGraphFit //
		///////////////////////
		template< typename Real , unsigned int Dim >
		QuadraticGraphFit< Real , Dim >::QuadraticGraphFit( Point< Real , Dim > p , Point< Real , Dim > n , const std::vector< Point< Real , Dim > > &nbrs )
			: LinearFit< Real , Dim >( p , n ) { _init( p , nbrs ); }

		template< typename Real , unsigned int Dim >
		QuadraticGraphFit< Real , Dim >::QuadraticGraphFit( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs )
			: LinearFit< Real , Dim >( p , nbrs ) { _init( p , nbrs ); }

		template< typename Real , unsigned int Dim >
		void QuadraticGraphFit< Real , Dim >::_init( Point< Real , Dim > p , const std::vector< Point< Real , Dim > > &nbrs )
		{
			Point< Real , Dim > tangents[Dim-1];

			// Get a set of Dim-1 linearly independent vectors spanning the tangent space
			for( unsigned int d=0 , i=0 ; d<Dim && i<Dim-1 ; d++ )
			{
				Point< Real , Dim > e;
				e[d] = (Real)1;
				if( fabs( Point< Real , Dim >::Dot( _l , e ) )>0.999 ) ;
				else tangents[i++] = e - Point< Real , Dim >::Dot( e , _l ) * _l;
			}

			// Perform G.S. orthogonalization
			for( unsigned int d=0 ; d<Dim-1 ; d++ )
			{
				// Make orthogonal to previous tangents
				for( unsigned int _d=0 ; _d<d ; _d++ ) tangents[d] -= Point< Real , Dim >::Dot( tangents[d] , tangents[_d] ) * tangents[_d];

				// For good measure (again) make orthogonal to the normal
				tangents[d] -= Point< Real , Dim >::Dot( tangents[d] , _l ) * _l;

				// Make unit length
				tangents[d] /= (Real)sqrt( tangents[d].squareNorm() );
			}

			// The projection onto the tangent space
			Matrix< Real , Dim , Dim-1 > P;
			for( unsigned int i=0 ; i<Dim-1 ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) P(j,i) = tangents[i][j];

			// Denote by {p} the outer product p \otimes p
			// Denote by [A] the representation of a symmetric matrix A as a vector

			// Given Q, for a point p':
			//	E(d=p'-p) = [ < d , n > - d^t * P^t * Q * P * d ]^2
			//            = [ d^t * n - [Q]^t * M * [{P * d}] ]^2
			// Given the set {p_i}, the error as a function of Q is:
			//	E(Q) = \sum_i [ d_i^t * n - [Q]^t * M * [{P * d_i}] ]^2
			//       = \sum_i [Q]^t * M * [{P * d_i}] * [{P * d_i}]^t * M^t * [Q] - 2 * [Q]^t * M * [{P * d_i}] * d_i^t * n
			// Taking the derivative with respect to Q and setting to zero:
			//     0 = \sum_i 2 * M * [{P * d_i}] * [{P * d_i}]^t * M^t * [Q] - 2 * M * [{P * d_i}] * d_i^t * n
			//     0 = \sum_i [{P * d_i}] * [{P * d_i}]^t * M^t * [Q] - [{P * d_i}] * d_i^t * n
			//   [Q] = - ( \sum_i [{P * d_i}] * [{P * d_i}]^t * M^t )^{-1} * ( \sum_i [{P * d_i}] * d_i^t * n )


			static const unsigned int _Dim = Dim-1;
			static const unsigned int _OuterProductDim = ( _Dim * ( _Dim+1 ) ) / 2;
			SquareMatrix< Real , _OuterProductDim > _M;
			for( unsigned int i=0 , idx=0 ; i<_Dim ; i++ ) for( unsigned int j=0 ; j<=i ; j++ , idx++ ) _M(idx,idx) = i==j ? 1 : 2;

			auto _OuterProduct = [&]( Point< Real , _Dim > q ) -> Point< Real , _OuterProductDim >
				{
					Point< Real , _OuterProductDim > op;
					for( unsigned int i=0 , idx=0 ; i<_Dim ; i++ ) for( unsigned int j=0 ; j<=i ; j++ , idx++ ) op[idx] = q[i] * q[j];
					return op;
				};

			auto _SymmetricMatrix = [&]( Point< Real , _OuterProductDim > q ) -> SquareMatrix< Real , _Dim >
				{
					SquareMatrix< Real , _Dim > A;
					for( unsigned int i=0 , idx=0 ; i<_Dim ; i++ ) for( unsigned int j=0 ; j<=i ; j++ , idx++ ) A(i,j) = A(j,i) = q[idx];
					return A;
				};

			SquareMatrix< Real , _OuterProductDim > A;
			Point< Real , _OuterProductDim > b;
			for( unsigned int i=0 ; i<nbrs.size() ; i++ )
			{
				Point< Real , Dim > d = nbrs[i] - p;
				Point< Real , _Dim > t = P * d;
				Point< Real , _OuterProductDim > _t = _OuterProduct(t);
				A += OuterProduct(_t,_t) * _M , b += _t * Point< Real , Dim >::Dot( d , _l );
			}
			_q = P.transpose() * _SymmetricMatrix( - A.inverse() * b ) * P;

			// Adjust the linear and constant terms
			// (q-p)^t * Q * (q-p) = q^t * Q * q - 2 q^t * Q * p + p^t * Q * p
			_l -= 2 * _q * p;
			_c += Point< Real , Dim >::Dot( p , _q * p );
		}

		template< typename Real , unsigned int Dim >
		void QuadraticGraphFit< Real , Dim >::Add( const QuadraticGraphFit &f ){ _q += f._q , _l += f._l ,  _c += f._c; }

		template< typename Real , unsigned int Dim >
		void QuadraticGraphFit< Real , Dim >::Scale( Real s ){ _q *=s , _l *=s , _c *=s; }

		/////////////////////////

		template< typename Real , unsigned Dim >
		std::vector< SquareMatrix< Real , Dim > > GetNormalCovariances( const std::vector< Point< Real , Dim > > &points , unsigned int nnNum , bool verbose )
		{
			std::vector< SquareMatrix< Real , Dim > > normalCovariances( points.size() );
			// Compute the kD-tree
			KDTree< Dim > kdTree( [&]( unsigned int idx ){ return points[idx]; } , (unsigned int)points.size() );

			ThreadPool::ParallelFor
				(
					0 , points.size() ,
					[&]( size_t i )
					{
						std::vector< std::pair< unsigned int , Point< Real , Dim > > > neighbors = kdTree.get_k_nearest_neighbors( points[i] , nnNum );
						std::vector< Point< Real , Dim > > nbrs( neighbors.size() );
						for( unsigned int j=0 ; j<neighbors.size() ; j++ ) nbrs[j] = neighbors[j].second;

						LinearFit< Real , Dim > lf( points[i] , nbrs );
						Point< Real , Dim > g = lf.gradient( points[i] );
						normalCovariances[i] = OuterProduct(g,g);
					}
				);
			return normalCovariances;
		}

	}
}
#endif // UNSIGNED_NORMALS_INCLUDED