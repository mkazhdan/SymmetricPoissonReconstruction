#ifndef HAT_INCLUDED
#define HAT_INCLUDED

#include <functional>
#include <Eigen/Sparse>
#include <Misha/Array.h>
#include <Misha/Geometry.h>
#include <Misha/Miscellany.h>
#include <Misha/Algebra.h>
#include <Misha/RegularGrid.h>
#include <Misha/RegularTree.h>
#include <Misha/Window.h>
#include "OrderedSamples.h"
#include "Samples.h"

namespace MishaK
{
	namespace Hat
	{

		//tex:
		// We use the symbol $\times$ to denote either the symmetric or alternating product of vectors
		// $$v \times w \equiv \frac{v\cdot w^\top \pm w\cdot v^\top}2,\qquad\forall v,w\in\mathbb{R}^k$$

		template< typename Real , unsigned int Dim , bool Sym >
		struct SquareMatrix : public InnerProductSpace< Real , SquareMatrix< Real , Dim , Sym > >
		{
			static const unsigned int Coefficients = Sym ? ( Dim * (Dim+1) ) / 2 : ( Dim * (Dim-1) ) / 2;

			// The zero matrix
			SquareMatrix( void ){}

			//tex:
			// The projection of a square matrix onto the space of symmetric/asymmetric matrices: $m\rightarrow \frac{ m + m^\top}2$.
			SquareMatrix( const MishaK::SquareMatrix< Real , Dim > m );

			//tex: The symmetric/asymmetric prouct of the vectors $v\times w$.
			SquareMatrix( Point< Real , Dim > v , Point< Real , Dim > w );

			MishaK::SquareMatrix< Real , Dim > operator()( void ) const;
			Real  operator[]( unsigned int idx ) const { return _p[idx]; }
			Real &operator[]( unsigned int idx ){ return _p[idx]; }

			void Add( const SquareMatrix &sym ){ _p += sym._p; }
			void Scale( Real s ){ _p *= s; }
			Real InnerProduct( const SquareMatrix &sym ) const { return MishaK::SquareMatrix< Real , Dim >::Dot( operator()() , sym() ); }

		protected:
			Point< Real , Coefficients > _p;
		};

		template< unsigned int Dim > using Index = typename RegularGrid< Dim >::Index;
		template< unsigned int Dim > using Range = typename RegularGrid< Dim >::Range;

		template< unsigned int Dim > struct ElementVector;

		// The representation of the restriction of the product of hat functions and derivatives to an element
		// Setting P(x) = x , Q(x) = (1-x) , dP(x) = 1 , dQ(x) = -1, this represents the function:
		//		P^p * Q^q * dP^dp * dQ^dq * coefficient
		template< unsigned int Dim >
		struct ElementFunction
		{
			ElementFunction( void ) : _coefficient(0) {}
			ElementFunction( Index< Dim > e , Index< Dim > i );

			ElementFunction d( unsigned int d ) const;

			ElementFunction &operator *= ( const ElementFunction &f );
			ElementFunction &operator *= ( double s );
			ElementFunction &operator /= ( double s ){ return operator *= ( 1./s); }
			ElementFunction  operator *  ( const ElementFunction &f ) const { ElementFunction _f = (*this) ; _f *= f ; return _f; }
			ElementFunction  operator *  ( double s ) const { ElementFunction f = (*this) ; f *= s ; return f; }
			ElementFunction  operator /  ( double s ) const { return operator * (1./s); }
			ElementFunction  operator - ( void ) const { return (*this) * (-1.); }

			double integral( unsigned int res ) const;
			double operator()( Point< double , Dim > p ) const;
			ElementVector< Dim > gradient( void ) const;

		protected:
			double _coefficient;
			Index< Dim > _p , _q , _d;
			friend std::ostream &operator << ( std::ostream &os , const ElementFunction &f )
			{
				return os << "( " << f._coefficient << " * x^" << f._p << " * (1-x)^" << f._q << " : d = " << f._d << " )";
			}
		};
		template< unsigned int Dim > ElementFunction< Dim > operator * ( double s , const ElementFunction< Dim > &f ){ return f*s; }

		template< unsigned int Dim >
		struct ElementVector
		{
			ElementVector( void ){}
			ElementFunction< Dim > &operator[]( unsigned int d ){ return _c[d]; }
			const ElementFunction< Dim > &operator[]( unsigned int d ) const { return _c[d]; }
			ElementVector &operator *= ( const ElementFunction< Dim > &f ){ for( unsigned int d=0 ; d<Dim ; d++ ) _c[d] *= f ; return *this; }
			ElementVector &operator *= ( double s ){ for( unsigned int d=0 ; d<Dim ; d++ ) _c[d] *= s ; return *this; }
			ElementVector &operator /= ( double s ){ return operator *= ( 1./s); }
			ElementVector  operator *  ( const ElementFunction< Dim > &f ) const { ElementVector v = (*this) ; v *= f ; return v; }
			ElementVector  operator *  ( double s ) const { ElementVector v = (*this) ; v *= s ; return v; }
			ElementVector  operator /  ( double s ) const { return operator * (1./s); }
			ElementVector  operator - ( void ) const { return (*this) * (-1.); }

			Point< double , Dim > operator()( Point< double , Dim > p ) const;

		protected:
			ElementFunction< Dim > _c[Dim];
			friend std::ostream &operator << ( std::ostream &os , const ElementVector &v )
			{
				os << "{ ";
				for( unsigned int d=0 ; d<Dim ; d++ )
				{
					if( d ) os << " , ";
					os << v._c[d];
				}
				os << " }";
				return os;
			}
		};
		template< unsigned int Dim > ElementVector< Dim > operator * ( double s , const ElementVector< Dim > &v ){ return v*s; }
		template< unsigned int Dim > ElementVector< Dim > operator * ( const ElementFunction< Dim > f , const ElementVector< Dim > &v ){ return v*f; }

		template< unsigned int Dim , bool Sym >
		struct ElementProduct
		{
			static const unsigned int Coefficients = SquareMatrix< double , Dim , Sym >::Coefficients;
			ElementProduct( void ){}
			ElementProduct( const ElementVector< Dim > &v1 , const ElementVector< Dim > &v2 );
			ElementFunction< Dim > &operator()( unsigned int i , unsigned int d ){ return _c[i][d]; }
			const ElementFunction< Dim > &operator()( unsigned int i , unsigned int d ) const { return _c[i][d]; }
			ElementProduct &operator *= ( const ElementFunction< Dim > &f ){ for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Coefficients ; d++ ) _c[i][d] *= f ; return *this; }
			ElementProduct &operator *= ( double s ){ for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Coefficients ; d++ ) _c[i][d] *= s ; return *this; }
			ElementProduct &operator /= ( double s ){ return operator *= ( 1./s); }
			ElementProduct  operator * ( const ElementFunction< Dim > &f ) const { ElementProduct w = (*this) ; w *= f ; return w; }
			ElementProduct  operator * ( double s ) const { ElementProduct w = (*this) ; w *= s ; return w; }
			ElementProduct  operator / ( double s ) const { return operator * (1./s); }
			ElementProduct  operator - ( void ) const { return (*this) * (-1.); }

			MishaK::SquareMatrix< double , Dim > operator()( Point< double , Dim > p ) const;

			static bool IsDiagonal( unsigned int i ){ return _IndexToCoefficients[i].first==_IndexToCoefficients[i].second; }

			struct IndexToCoefficients
			{
				IndexToCoefficients( void );
				std::pair< unsigned int , unsigned int > coefficients[Coefficients];
				std::pair< unsigned int , unsigned int > &operator[]( unsigned int idx ){ return coefficients[idx]; }
				const std::pair< unsigned int , unsigned int > &operator[]( unsigned int idx ) const { return coefficients[idx]; }
			};

		protected:
			static IndexToCoefficients _IndexToCoefficients;
			ElementFunction< Dim > _c[2][Coefficients];
		};

		template< unsigned int Dim > struct ScalarFunctions;
		template< unsigned int Dim , bool Sym > struct ProductFunctions;


		// The base case (when Dim=1)
		struct _Basis
		{
			// The range of elements in the support of the i-th function
			static std::pair< int , int > FunctionSupport( int i );
			static std::pair< int , int > FunctionSupport( std::pair< int , int > r );

			// The range of functions supported on the i-th element
			static std::pair< int , int >  ElementSupport( int i );
			static std::pair< int , int >  ElementSupport( std::pair< int , int > r );

			// The integral of x^p * (1-x)^q over the interval
			static double Integral( unsigned int p , unsigned int q );
		};

		template< unsigned int Dim >
		struct Basis
		{
			// The range of elements in the support of a hat basis function
			static Range< Dim > FunctionSupport( Index< Dim > f );
			static Range< Dim > FunctionSupport( Range< Dim > r );

			// The range of hat basis functions supported on an element
			static Range< Dim > ElementSupport( Index< Dim > e );
			static Range< Dim > ElementSupport( Range< Dim > r );

			// The integral of the product of two hat functions over an element
			static double ElementMass( unsigned int res , Index< Dim > e , Index< Dim > i , Index< Dim > j );

			// The integral of the dot-product of the gradients of two hat functions over an element
			static double ElementStiffness( unsigned int res , Index< Dim > e , Index< Dim > i , Index< Dim > j );

			// The integral of the dot-product of the product of the gradients of two pairs of hat functions over an element
			template< bool Sym >
			static double ElementGradientProductMass( unsigned int res , Index< Dim > e , std::pair< Index< Dim > , Index< Dim > > i , std::pair< Index< Dim > , Index< Dim > > j );

			// Estimates the integral over the element
			template< typename T , typename F /* = std::function< T ( Point< double , Dim > ) */ >
			static T Integral( unsigned int res , Index< Dim > e , F f , unsigned samplingRes );

			// Estimates the integral over the entire domain
			template< typename T , typename F /* = std::function< T ( Point< double , Dim > ) */ >
			static T Integral( F f , unsigned samplingRes );

			// Get the function associated with an index
			static std::function< double ( Point< double , Dim > ) > Scalar( unsigned int res , Index< Dim > i );

			// Get the gradient of the function associated with an index
			static std::function< Point< double , Dim > ( Point< double , Dim > ) > Gradient( unsigned int res , Index< Dim > i );

		protected:
			template< unsigned int Radius >
			static int _RelativeIndex( Index< Dim > e , Index< Dim > f );

			template< unsigned int Radius , unsigned int D >
			static void _RelativeIndex( Index< Dim > e , Index< Dim > f , unsigned int &idx );

			template< unsigned int Radius >
			static constexpr unsigned int _Width( void ){ return 2*Radius+2; }

			friend struct  ScalarFunctions< Dim >;
			friend struct ProductFunctions< Dim , true  >;
			friend struct ProductFunctions< Dim , false >;
		};

		// A type acting as a statically allocated multi-dimensional array
		// (though the values are actually allocated on the heap)
		template< typename T , unsigned int ... Widths > struct Stencil;

		template< typename T , unsigned int Width >
		struct Stencil< T , Width > : public VectorSpace< double , Stencil< T , Width > >
		{
			Stencil( void );
			Stencil( const Stencil &stencil );
			Stencil( Stencil &&stencil );
			~Stencil( void );

			Stencil &operator = ( const Stencil &stencil );
			Stencil &operator = ( Stencil &&stencil );

			T &operator[]( unsigned int idx ){ return _values[idx]; }
			const T &operator[]( unsigned int idx ) const { return _values[idx]; }

			T &operator()( Index< 1 > i ){ return _values[ i[0] ]; }
			const T &operator()( Index< 1 > i ) const { return _values[ i[0] ]; }

			void Add( const Stencil &s );
			void Scale( double s );
		protected:
			T &_value( int *idx ){ return _values[ idx[0] ]; }
			const T &_value( int *idx ) const { return _values[ idx[0] ]; }

			Pointer( T ) _values;

			template< typename _T , unsigned int ... _Widths > friend struct Stencil;
		};

		template< typename T , unsigned int Width , unsigned int ... Widths >
		struct Stencil< T , Width , Widths... > : public VectorSpace< double , Stencil< T , Width , Widths... > >
		{
			using SubStencil = Stencil< T , Widths ... >;

			Stencil( void );
			Stencil( const Stencil &stencil );
			Stencil( Stencil &&stencil );
			~Stencil( void );

			Stencil &operator = ( const Stencil &stencil );
			Stencil &operator = ( Stencil &&stencil );

			SubStencil &operator[]( unsigned int idx ){ return _values[idx]; }
			const SubStencil &operator[]( unsigned int idx ) const { return _values[idx]; }

			T &operator()( Index< 1+sizeof...(Widths) > i ){ return _value( &i[0] ); }
			const T &operator()( Index< 1+sizeof...(Widths) > i ) const { return _value( &i[0] ); }


			void Add( const Stencil &s );
			void Scale( double s );
		protected:
			Pointer( SubStencil ) _values;

			T &_value( int *idx ){ return _values[ idx[0] ]._value( idx+1 ); }
			const T &_value( int *idx ) const { return _values[ idx[0] ]._value( idx+1 ); }

			template< typename _T , unsigned int ... _Widths > friend struct Stencil;
		};

		template< typename T , unsigned int Dim , unsigned int Width , unsigned int ... Widths > struct _Stencil;

		template< typename T , unsigned int Width , unsigned int ... Widths >
		struct _Stencil< T , 1 , Width , Widths... >{ using type = Stencil< T , Width , Widths... >; };

		template< typename T , unsigned int Dim , unsigned int Width , unsigned int ... Widths >
		struct _Stencil{ using type = typename _Stencil< T , Dim-1 , Width , Width , Widths... >::type; };

		// A stencil with the same resolution across all dimensions
		template< typename T , unsigned int Dim , unsigned int Width >
		using SquareStencil = typename _Stencil< T , Dim , Width >::type;

		// An iterator allowing for triplets to be set by multiple threads, without write-on-write conflicts, but still behaving as a single container
		template< typename T >
		struct ThreadPoolTriplets
		{
			struct iterator
			{
				unsigned int thread, index;
				iterator& operator++(void) { if (index + 1 == _triplets[thread].size()) _nextThread(), index = 0; else index++; return *this; }
				iterator  operator++(int) { iterator iter = *this; if (index + 1 == _triplets[thread].size()) _nextThread(), index = 0; else index++; return iter; }
				bool operator == (iterator iter) const { return thread == iter.thread && index == iter.index; };
				bool operator != (iterator iter) const { return thread != iter.thread || index != iter.index; };
				Eigen::Triplet< T >& operator * (void) { return _triplets[thread][index]; }
				Eigen::Triplet< T >* operator -> (void) { return &(_triplets[thread][index]); }
				const Eigen::Triplet< T >& operator *  (void) const { return _triplets[thread][index]; }
				const Eigen::Triplet< T >* operator -> (void) const { return &(_triplets[thread][index]); }
			protected:
				ThreadPoolTriplets& _triplets;

				iterator( ThreadPoolTriplets& ompTriplets , bool begin ) : _triplets(ompTriplets), index(0)
				{
					if (begin)
					{
						thread = -1;
						_nextThread();
					}
					else thread = (unsigned int)_triplets.size();
				}

				void _nextThread(void)
				{
					do { thread++; } while (thread < _triplets.size() && _triplets[thread].size() == 0);
				}

				friend ThreadPoolTriplets;
			};

			struct const_iterator
			{
				unsigned int thread, index;
				const_iterator& operator++(void) { if (index + 1 == _triplets[thread].size()) _nextThread(), index = 0; else index++; return *this; }
				const_iterator  operator++(int) { const_iterator iter = *this; if (index + 1 == _triplets[thread].size()) _nextThread(), index = 0; else index++; return iter; }
				bool operator == (const_iterator iter) const { return thread == iter.thread && index == iter.index; };
				bool operator != (const_iterator iter) const { return thread != iter.thread || index != iter.index; };
				const Eigen::Triplet< T >& operator *  (void) const { return _triplets[thread][index]; }
				const Eigen::Triplet< T >* operator -> (void) const { return &(_triplets[thread][index]); }
			protected:
				const ThreadPoolTriplets& _triplets;

				const_iterator(const ThreadPoolTriplets& ompTriplets, bool begin) : _triplets(ompTriplets), index(0)
				{
					if (begin)
					{
						thread = -1;
						_nextThread();
					}
					else thread = (unsigned int)_triplets.size();
				}

				void _nextThread(void)
				{
					do { thread++; } while (thread < _triplets.size() && _triplets[thread].size() == 0);
				}

				friend ThreadPoolTriplets;
			};

			ThreadPoolTriplets(void) : _triplets( ThreadPool::NumThreads() ) {}
			std::vector< std::vector< Eigen::Triplet< T > > > _triplets;
			std::vector< Eigen::Triplet< T > >& operator[] (unsigned int thread) { return _triplets[thread]; }
			const std::vector< Eigen::Triplet< T > >& operator[] (unsigned int thread) const { return _triplets[thread]; }
			void clear(void)
			{
				for (unsigned int i = 0; i < _triplets.size(); i++)
				{
					std::vector< Eigen::Triplet< T > > t;
					std::swap(_triplets[i], t);
				}
			}
			size_t size(void) const { return _triplets.size(); }
			iterator begin(void) { return iterator(*this, true); }
			iterator end(void) { return iterator(*this, false); }
		};

		// An iterator allowing for triplets to be set by multiple threads, without write-on-write conflicts, but still behaving as a single container
		template< typename T >
		struct MultiThreadedTriplets
		{
			struct iterator
			{
				unsigned int thread , index;
				iterator& operator++( void ) { if( index+1==_mtTriplets[thread].size() ) _nextThread() , index = 0 ; else index++ ; return *this; }
				iterator  operator++( int ) { iterator iter = *this ; if( index+1==_mtTriplets[thread].size() ) _nextThread() , index = 0 ; else index++ ; return iter; }
				bool operator == ( iterator iter ) const { return thread == iter.thread && index == iter.index; };
				bool operator != ( iterator iter ) const { return thread != iter.thread || index != iter.index; };
				Eigen::Triplet< T >& operator * (void) { return _mtTriplets[thread][index]; }
				Eigen::Triplet< T >* operator -> (void) { return &( _mtTriplets[thread][index] ); }
				const Eigen::Triplet< T >& operator *  (void) const { return _mtTriplets[thread][index]; }
				const Eigen::Triplet< T >* operator -> (void) const { return &( _mtTriplets[thread][index] ); }
			protected:
				MultiThreadedTriplets& _mtTriplets;

				iterator( MultiThreadedTriplets& mtTriplets , bool begin ) : _mtTriplets( mtTriplets ), index(0)
				{
					if( begin )
					{
						thread = -1;
						_nextThread();
					}
					else thread = (unsigned int)_mtTriplets.size();
				}

				void _nextThread( void )
				{
					do { thread++; } while( thread<_mtTriplets.size() && _mtTriplets[thread].size()==0 );
				}

				friend MultiThreadedTriplets;
			};

			struct const_iterator
			{
				unsigned int thread, index;
				const_iterator& operator++( void ) { if( index+1==_mtTriplets[thread].size() ) _nextThread() , index = 0 ; else index++ ; return *this; }
				const_iterator  operator++( int ) { const_iterator iter = *this ; if( index+1==_mtTriplets[thread].size() ) _nextThread() , index = 0 ; else index++ ; return iter; }
				bool operator == ( const_iterator iter ) const { return thread == iter.thread && index == iter.index; };
				bool operator != ( const_iterator iter ) const { return thread != iter.thread || index != iter.index; };
				const Eigen::Triplet< T >& operator *  ( void ) const { return _mtTriplets[thread][index]; }
				const Eigen::Triplet< T >* operator -> ( void ) const { return &( _mtTriplets[thread][index] ); }
			protected:
				const MultiThreadedTriplets& _mtTriplets;

				const_iterator( const MultiThreadedTriplets& mtTriplets , bool begin ) : _mtTriplets( mtTriplets ) , index(0)
				{
					if( begin )
					{
						thread = -1;
						_nextThread();
					}
					else thread = (unsigned int)_mtTriplets.size();
				}

				void _nextThread( void )
				{
					do { thread++; } while( thread<_mtTriplets.size() && _mtTriplets[thread].size()==0 );
				}

				friend MultiThreadedTriplets;
			};

			MultiThreadedTriplets( void ) : _triplets( ThreadPool::NumThreads() ){}
			std::vector< std::vector< Eigen::Triplet< T > > > _triplets;
			std::vector< Eigen::Triplet< T > >& operator[] ( unsigned int thread ) { return _triplets[thread]; }
			const std::vector< Eigen::Triplet< T > >& operator[] ( unsigned int thread ) const { return _triplets[thread]; }
			void clear( void )
			{
				for( unsigned int i= 0 ; i<_triplets.size() ; i++ )
				{
					std::vector< Eigen::Triplet< T > > t;
					std::swap( _triplets[i] , t );
				}
			}
			size_t size( void ) const { return _triplets.size(); }
			iterator begin( void ) { return iterator( *this , true ); }
			iterator end( void ) { return iterator( *this , false ); }
		};

#include "Hat.inl"
#include "Hat.adaptive.h"
#include "Hat.scalars.h"
#include "Hat.products.h"
	}
}
#endif // HAT_INCLUDED