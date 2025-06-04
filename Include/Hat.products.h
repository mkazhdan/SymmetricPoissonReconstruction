//tex:
// The space is spanned by $\{\omega_I\}$ where $I=(i,j)$ with $i\leq j$ (resp. $i < j$)
// index pairs of functions with overlapping support and $\omega_I = \frac{\nabla\phi_i\otimes\nabla\phi_j\pm\nabla\phi_j\otimes\nabla\phi_i}2$. //
// Given coefficients $X\in\mathbb{R}^N$, we denote by the associated function by $$\omega_X\equiv\sum_{I=1}^N x_I\cdot\omega_I.$$ 
template< unsigned int Dim , bool Sym >
struct ProductFunctions
{
	template< typename T , unsigned int Rank , unsigned int Radius >
	using IntegrationStencil = typename ScalarFunctions< Dim >::template IntegrationStencil< T , 2*Rank , Radius >;

	template< typename T >
	struct FullIntegrationStencil
	{
		struct Entry
		{
			Index< Dim > G1 , G2;	// All pairs of overlapping functions (except g1=g2 if not symmetric)
			T value;
			unsigned int _g1, _g2; 
			Entry( Index< Dim > G1 , Index< Dim > G2 , T value ) : G1(G1) , G2(G2) , value(value) 
			{
				for( unsigned int d=0 ; d<Dim ; d++ ) G1[d]++ , G2[d]++;
				_g1 = Window::IsotropicGetIndex< Dim , 3 >( &G1[0] );
				_g2 = Window::IsotropicGetIndex< Dim , 3 >( &G2[0] );
			}
		};

		struct Row
		{
			Index< Dim > F2;	// All functions with overlapping support (except f1=f2 if not symmetric)
			std::vector< Entry > entries;
			unsigned int _f2;
			Row( Index< Dim > F2 ) : F2(F2)
			{
				for( unsigned int d=0 ; d<Dim ; d++ ) F2[d]++;
				_f2 = Window::IsotropicGetIndex< Dim , 3 >( &F2[0] );
			}

			unsigned int end0_ , end_0;
		};

		FullIntegrationStencil( const IntegrationStencil< T , 2 , 0 > &stencil , unsigned int res );
		const std::vector< Row > &rows( Index< Dim > f1 ) const;
		std::vector< std::vector< Row > > &rows( void ){ return _rows; }

	protected:
		unsigned int _res;
		std::vector< std::vector< Row > > _rows;
	};

	// Function for combining a values from a pair of scalar functions to get the coefficient of the product
	static double Coefficient( std::pair< double , double > f1 , std::pair< double , double > f2 , size_t i1 , size_t i2 );


	ProductFunctions( unsigned int resolution );
	size_t resolution( void ) const { return _r; }

	// The number of product functions
	size_t functionNum( void ) const { return _matrixInfo.entries(false); }

	// Returns the index associated to a pair vertices and sets the flag if the orientation is reversed
	// Throws an exception if the pair does not index a basis
	size_t index( Index< Dim > F1 , Index< Dim > F2 , bool &flip ) const;
	size_t index( std::pair< size_t , size_t > idx , bool &flip ) const { return index( ScalarFunctions< Dim >::FunctionIndex(idx.first,_r) , ScalarFunctions< Dim >::FunctionIndex(idx.second,_r) , flip ); }

	// Sets the index associated to a pair vertices and sets the flag if the orientation is reversed
	// Returns false if the pair does not index a basis
	bool setIndex( Index< Dim > F1 , Index< Dim > F2 , size_t &i , bool &flip ) const;
	bool setIndex( std::pair< size_t , size_t > idx , size_t &i , bool &flip ) const { return setIndex( ScalarFunctions< Dim >::FunctionIndex(idx.first,_r) , ScalarFunctions< Dim >::FunctionIndex(idx.second,_r) , i , flip ); }

	// Returns the index pairs associated with the functions
	std::vector< std::pair< Index< Dim > , Index< Dim > > > indices( void ) const;

	//////////////
	// Products //
	//////////////

	//tex: Computes the product coefficients, should satisfy $$\omega_{x \times y} = \nabla\phi_x\times\nabla\phi_y.$$
	Eigen::VectorXd product( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

	//tex: Computes the transpose of the product operator associated to $x\in\mathbb{R}^n$ applied to $Z\in\mathbb{R}^N$:
	// $$(x\times\cdot)^\top\cdot Z.$$
	Eigen::VectorXd productTranspose( const Eigen::VectorXd &x , const Eigen::VectorXd &Z ) const;

	// Returns the function corresponding to g -> df * dg
	Eigen::SparseMatrix< double > product( const Eigen::VectorXd &f ) const;


	////////////////////////////////////////////
	// Evaluation and Monte-Carlo integration //
	////////////////////////////////////////////

	// Evaluation of a function's value, represented in the product function basis
	SquareMatrix< double , Dim , Sym > value( const Eigen::VectorXd &xy , Point< double , Dim > p ) const;

	// Estimates the dot-product of two scalar functions
	template< typename F1 /* = std::function< SquareMatrix< double , Dim , Sym > ( Point< double , Dim > ) > */ , typename F2 /* = std::function< SquareMatrix< double , Dim , Sym > ( Point< double , Dim > ) > */ >
	static double TensorDotProduct( F1 f1 , F2 f2 , unsigned int samplesPerDimension ){ return Basis< Dim >::template Integral< double >( [&]( Point< double , Dim > p ){ return SquareMatrix< double , Dim , Sym >::Dot( f1(p) , f2(p) ); } , samplesPerDimension ); }


	//////////////////////////////////////////
	// Single function integration stencils //
	//////////////////////////////////////////

	static IntegrationStencil< SquareMatrix< double , Dim , Sym > , 1 , 0 > ValueStencil( unsigned int r );

	////////////////////////////////////////////
	// Pair of functions integration stencils //
	////////////////////////////////////////////

	static IntegrationStencil< double , 2 , 0 > MassStencil( unsigned int r );

	////////////////////////
	// Stencil evaluation //
	////////////////////////

	// Evaluates the stencil on the pair of functions generated as the product of pairs of functions
	template< typename T , typename Indexer /* = Hat::BaseIndex< Dim > */ >
	T operator()( const Indexer &indexer , const FullIntegrationStencil< T > &stencil , const Eigen::VectorXd &x1 , const Eigen::VectorXd &x2 , const Eigen::VectorXd &y1 , const Eigen::VectorXd &y2 ) const;

	// Transforms a stencil into a linear operator
	Eigen::SparseMatrix< double > systemMatrix( IntegrationStencil< double , 2 , 0 > stencil ) const;


	//////////////////////////
	// Dual representations //
	//////////////////////////

	// Integrates the basis functions against a piecewise constant tensor field
	template< typename PiecewiseConstantTensorField /* = std::function< SquareMatrix< double , Dim , Sym > ) ( size_t e ) > */ >
	Eigen::VectorXd valueDual( PiecewiseConstantTensorField T ) const;

	// Integrates the basis functions against the tensor field with the given coefficients
	Eigen::VectorXd valueDual( const Eigen::VectorXd &xy ) const;

	//////////////////////////////
	// Bilinear representations //
	//////////////////////////////

	// Returns the mass matrix for the product functions
	Eigen::SparseMatrix< double > mass( void ) const { return systemMatrix( MassStencil( _r ) ); }


	//tex: Given a vector $Z\in\mathbb{R}^N$, if we denote by $\mathbf{A}_Z\in\mathbb{R}^{n\times n}$ the representation of $Z$ as a matrix,
	// then $y^\top\cdot\mathbf{A}_Z\cdot x = Z^\top\cdot(x\times y)$.
	Eigen::SparseMatrix< double > toMatrix( const Eigen::VectorXd &xy ) const;
	Eigen::VectorXd toVector( const Eigen::SparseMatrix< double > &XY ) const;

protected:

	unsigned int _r;

	typename ScalarFunctions< Dim >::template MatrixInfo< 1 , Sym > _matrixInfo;
};

#include "Hat.products.inl"