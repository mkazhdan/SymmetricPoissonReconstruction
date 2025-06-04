//tex:
// The space is spanned by $\{\phi_i\}$ where $i$ indexes the corners of the grids and $\phi_i$ is the ``hat'' function.
// Given coefficients $x\in\mathbb{R}^n$, we denote by the associated function by $$\phi_x\equiv\sum_{i=1}^n x_i\cdot\phi_i.$$ 

template< unsigned int Dim > struct ScalarFunctions;

// A specialization of the Dim=0 case to allow boundary system matrices
template<> struct ScalarFunctions< 0 >
{
	template< typename T , unsigned int Rank , unsigned int Radius > struct IntegrationStencil{};
	template< unsigned int Radius > static constexpr unsigned int StencilSize( void ){ return 1; }
	template< unsigned int Radius > struct StencilCaseTable{ static constexpr unsigned int Cases( void ){ return 1; } };
};


template< unsigned int Dim >
struct ScalarFunctions
{
	template< unsigned int Radius > static constexpr unsigned int StencilWidth( void ){ return 2+2*Radius; }
	template< unsigned int Radius > static constexpr unsigned int StencilSize( void ){ return ScalarFunctions< Dim-1 >::template StencilSize< Radius >() * StencilWidth< Radius >(); }

	// A structure describing a stencil case table
	template< unsigned int Radius >
	struct StencilCaseTable
	{
		static const unsigned int Width = 2*Radius + 1;

		// The maximum number of cases
		static constexpr unsigned int Cases( void ){ return ScalarFunctions< Dim-1 >::template StencilCaseTable< Radius >::Cases() * Width; }

		// The range of cases supported
		static Hat::Range< Dim > Range( unsigned int res );

		static Index< Dim > IndexToCase( Index< Dim > I , unsigned int res );
		static Index< Dim > CaseToIndex( Index< Dim > C , unsigned int res );

		// The number of indices associated to this case
		static size_t Size( Hat::Index< Dim > C , unsigned int res );

		// The index within a scase
		static size_t SubIndex( Hat::Index< Dim > I , unsigned int res );
	};

	// A structure for indexing the entries of a symmetric/anti-symmetric matrix
	template< unsigned int Radius , bool Sym >
	struct MatrixInfo
	{
		MatrixInfo( unsigned int res );

		// Given a pair of function indices, with f2 in the Radius-ring of f1 and f1<(=)f2, return the index of the associated matrix entry
		size_t entry( Index< Dim > f1 , Index< Dim > f2 ) const;

		// The total number of entries in the matrix
		size_t entries( bool all ) const;

		// Process the upper triangular entries in the specified row
		template< typename F /* = std::function< void ( Index< Dim > , size_t ) > */ >
		void process( Index< Dim > I , F f ) const;

		// Process all the entries in the specified row
		template< typename F /* = std::function< void ( Index< Dim > , size_t , bool ) > */ >
		void processAll( Index< Dim > I , F f ) const;

		// Returns the row size
		size_t entries( Index< Dim > I , bool all ) const;

	protected:
		struct _Info
		{
			SquareStencil< size_t , Dim , StencilCaseTable< Radius >::Width > stencil;
			size_t sizes[2],  offset;
		};

		unsigned int _res;
		Index< Dim > _off;
		Range< Dim > _caseRange , _infoRange , _fRange;
		SquareStencil< _Info , Dim , StencilCaseTable< Radius >::Width > _info;
	};

	struct ProlongationStencil : public SquareStencil< double , Dim , StencilCaseTable<1>::Width >
	{
		ProlongationStencil( void );
		Range< Dim > range( void ) const { return _range; }
	protected:
		Range< Dim > _range;
	};

	struct Prolongation
	{
		template< typename ProlongationFunctor/*=std::function< void ( Index< Dim > F_fine , double weight ) >*/ >
		static void Process( Index< Dim > F_coarse , unsigned int coarseRes , ProlongationFunctor pFunctor );
	};

	struct Restriction
	{
		template< typename RestrictionFunctor/*=std::function< void ( Index< Dim > F_coarse , double weight ) >*/ >
		static void Process( Index< Dim > F_fine , RestrictionFunctor rFunctor );
	protected:
		template< unsigned int D , typename RestrictionFunctor/*=std::function< void ( Index< Dim > F_coarse , double weight ) >*/ >
		static void _Process( Index< Dim > F_fine , Index< Dim > F_coarse , double weight , RestrictionFunctor &rFunctor );
	};

	// A struct storing the integral of all Rank-tuples of functions over an element
	// In addition to storing the values, this provides an operator for evaluating the integral of a Rank-tuple of functions over (spans of) elements(s)
	template< typename T , unsigned int Rank , unsigned int Radius >
	struct IntegrationStencil : public SquareStencil< T , Rank , StencilSize< Radius >() >
	{
		using SquareStencil< T , Rank , StencilSize< Radius >() >::_values;

		template< typename ... F > T operator()( Index< Dim > e , F ... f ) const;
		template< typename ... F > T operator()( Range< Dim > eRange , F ... f ) const;
	};

	template< typename T , unsigned int Radius >
	struct FullIntegrationStencil
	{
		static constexpr unsigned int StencilNum( void ); // (3+4*Radius)^Dim
		using Entry = std::tuple< Index< Dim > , size_t , T >;
		using Row = std::vector< Entry >;

		FullIntegrationStencil( void );
		FullIntegrationStencil( const IntegrationStencil< T , 2 , Radius > &stencil , unsigned int res );
		const Row &row( Index< Dim > F ) const;
		static unsigned int StencilIndex( Index< Dim > F , unsigned int res );

		friend ScalarFunctions;
	protected:
		unsigned int _res;
		std::vector< Row > _rows;
	};

	template< typename T >
	static FullIntegrationStencil< T , 0 > Restrict( const FullIntegrationStencil< T , 0 > &stencil );

	
	ScalarFunctions( unsigned int resolution ) : _r(resolution) , _eRange( ElementRange(resolution) ) , _fRange( FunctionRange(resolution) ){}
	unsigned int resolution( void ) const { return _r; }

	// The number of scalar functions
	static size_t FunctionNum( unsigned int res );
	size_t        functionNum( void ) const { return FunctionNum( _r ); }

	// The number of elements/cells in the grid
	static size_t ElementNum( unsigned int res );
	size_t        elementNum( void ) const { return ElementNum( _r ); }

	// Conversion from multi-dimensional element index to linearized element index
	static size_t       ElementIndex( Index< Dim > i , unsigned int res );
	size_t              elementIndex( Index< Dim > i ) const { return ElementIndex( i , _r ); }
	// Conversion from linearized element index to multi-dimensional element index
	static Index< Dim > ElementIndex( size_t i , unsigned int res );
	Index< Dim >        elementIndex( size_t i ) const { return ElementIndex( i , _r ); }

	// Conversion from multi-dimensional function index to linearized function index
	static size_t       FunctionIndex( Index< Dim > i , unsigned int res );
	size_t              functionIndex( Index< Dim > i ) const { return FunctionIndex( i , _r ); }
	// Conversion from linearized function index to multi-dimensional function index
	static Index< Dim > FunctionIndex( size_t i , unsigned int res );
	Index< Dim >        functionIndex( size_t i ) const { return FunctionIndex( i , _r ); }

	// The range of elements
	static Range< Dim > ElementRange( unsigned int res );
	Range< Dim >        elementRange( void ) const { return _eRange; }
	// The range of functions
	static Range< Dim > FunctionRange( unsigned int res );
	Range< Dim >        functionRange( void ) const { return _fRange; }


	////////////////////////////////////////////
	// Evaluation and Monte-Carlo integration //
	////////////////////////////////////////////
	
	// Evaluation of a function's value, represented in the hat basis
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	double value( const Indexer & indexer , const Eigen::VectorXd &x , Point< double , Dim > p , unsigned int thread ) const;

	// Evaluation of a function's gradient, represented in the hat basis
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Point< double , Dim > gradient( const Indexer & indexer , const Eigen::VectorXd &x , Point< double , Dim > p , unsigned int thread ) const;

	// Estimates the dot-product of two scalar functions
	template< typename F1 /* = std::function< double ( Point< double , Dim > ) > */ , typename F2 /* = std::function< double ( Point< double , Dim > ) > */ >
	static double ScalarDotProduct( F1 f1 , F2 f2 , unsigned int samplesPerDimension ){ return Basis< Dim >::template Integral< double >( [&]( Point< double , Dim > p ){ return f1(p)*f2(p); } , samplesPerDimension ); }
	// Estimates the dot-product of two vector functions
	template< typename V1 /* = std::function< Point< double , Dim > ( Point< double , Dim > ) > */ , typename V2 /* = std::function< Point< double , Dim > ( Point< double , Dim > ) > */ >
	static double VectorDotProduct( V1 v1 , V2 v2 , unsigned int samplesPerDimension ){ return Basis< Dim >::template Integral< double >( [&]( Point< double , Dim > p ){ return Point< double , Dim >::Dot( v1(p) , v2(p) ); } , samplesPerDimension ); }

	//////////////////////////////////////////
	// Single function integration stencils //
	//////////////////////////////////////////
	
	// For a function supported on an element, returns the integral of the funciton over the element
	static IntegrationStencil< double                , 1 , 0 >             ValueStencil( unsigned int r );
	// For a function supported on an element, returns the integral of the partial derivatives over the element
	static IntegrationStencil< Point< double , Dim > , 1 , 0 > PartialDerivativeStencil( unsigned int r );

	////////////////////////////////////////////
	// Pair of functions integration stencils //
	////////////////////////////////////////////
	
	// For a pair of functions supported on an element, returns the integral of the product of the functions over the element
	static IntegrationStencil< double                         , 2 , 0 >          MassStencil( unsigned int r );
	// For a pair of functions supported on an element, returns the integral of the dot-product of the gradients of the functions over the element
	static IntegrationStencil< double                         , 2 , 0 >     StiffnessStencil( unsigned int r );
	// For a pair of functions supported on an element, returns the integral of the products of the partial derivatives over the element
	static IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 2 , 0 > ConstantWeightedStiffnessStencil( unsigned int r );
	// For a triplet of functions supported on an element, returns the integral of the products of the partial derivatives of the first two, with the coefficients of the third over the element
	static IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 3 , 0 > LinearWeightedStiffnessStencil( unsigned int r );

	////////////////////////
	// Stencil evaluation //
	////////////////////////
	
	// Evaluates a stencil on a pair of functions
	template< typename T , typename Indexer /* = Hat::Indexer< Dim > */ >
	T operator()( const Indexer & indexer , const IntegrationStencil< T , 2 , 0 > &stencil , const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

	// Evaluates a full stencil on a pair of functions
	template< typename T , typename Indexer /* = Hat::Indexer< Dim > */ >
	T operator()( const Indexer & indexer , const FullIntegrationStencil< T , 0 > &stencil , const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

	// Evaluates a stencil on a function
	template< typename T , typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::Matrix< T , 1 , Eigen::Dynamic > operator()( const Indexer & indexer , const IntegrationStencil< T , 2 , 0 > &stencil , const Eigen::VectorXd &x ) const;

	// Evaluates a full stencil on a function
	template< typename T , typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::Matrix< T , 1 , Eigen::Dynamic > operator()( const Indexer & indexer , const FullIntegrationStencil< T , 0 > &stencil , const Eigen::VectorXd &x ) const;

	// Transforms a stencil into a linear operator
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > systemMatrix( const Indexer & indexer , IntegrationStencil< double , 2 , 0 > stencil ) const;

	// Transforms a full stencil into a linear operator
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > systemMatrix( const Indexer & indexer , FullIntegrationStencil< double , 0 > stencil ) const;

	// Transforms a full stencil into a linear operator
	// [WARNING] The stencil is assumed to be symmetric/isotropic
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > boundarySystemMatrix( const Indexer & indexer , typename ScalarFunctions< Dim-1 >::template IntegrationStencil< double , 2 , 0 > stencil ) const;

	//////////////////////////
	// Dual representations //
	//////////////////////////

	// Integrates the basis functions against a piecewise constant scalar field
	//tex:
	// $$d[i] = \int\phi_i\cdot f\,dp$$
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename PiecewiseConstantScalarField /* = std::function< double ( Hat::Index< Dim > E ) > */ >
	Eigen::VectorXd valueDual( const Indexer & indexer , PiecewiseConstantScalarField f ) const;

	// Integrates the basis functions against a piecewise-linear scalar field with the given coefficients
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::VectorXd valueDual( const Indexer & indexer , const Eigen::VectorXd &x ) const;

	// Integrates the gradients of the basis functions against a piecewise constant vector field
	//tex:
	// $$d[i] = \int\langle\nabla\phi_i,\vec{v}\rangle\,dp$$
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename PiecewiseConstantVectorField /* = std::function< Point< double , Dim > ( Hat::Index< Dim > E ) > */ >
	Eigen::VectorXd gradientDual( const Indexer & indexer , PiecewiseConstantVectorField VF ) const;

	// Integrates the gradients of the basis against the gradients of a piecewise-linear scalar field with the given coefficients
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::VectorXd gradientDual( const Indexer & indexer , const Eigen::VectorXd &x ) const;

	// Integrates the basis functions against a weighted sum of delta functions
	template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , Data > ( unsigned int idx ) > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
	std::vector< Data > deltaDual( const Indexer &indexer , SampleFunctor F , size_t sampleNum , WeightFunctor wF ) const;
	template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
	std::vector< Data > deltaDual( const Indexer &indexer , const std::vector< std::pair< Point< double , Dim > , Data > > &samples , WeightFunctor wF ) const;
	template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , Data > ( unsigned int idx ) > */ >
	std::vector< Data > deltaDual( const Indexer &indexer , SampleFunctor F , size_t sampleNum ) const;
	template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ >
	std::vector< Data > deltaDual( const Indexer &indexer , const std::vector< std::pair< Point< double , Dim > , Data > > &samples ) const;

	//////////////////////////////
	// Bilinear representations //
	//////////////////////////////

	//tex:
	// $$M(x,y) = \sum_{e\in E} x(p)\cdot y(p)\ dp$$ 

	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > mass( const Indexer & indexer ) const;

	// Computes the mass matrix w.r.t. to the piecewise-constant inner-product given by the scalar values
	//tex:
	// $$M(x,y) = \sum_{e\in E}\int_e w[e]\cdot x(p)\cdot y(p)\ dp$$ 

	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename MassFunctor /* = std::function< double ( Index< Dim > E ) > */ >
	Eigen::SparseMatrix< double > mass( const Indexer & indexer , MassFunctor && m ) const;

	//tex:
	// $$S(x,y) = \sum_{e\in E}\int_e \langle\nabla x|_p,\nabla y|_p\rangle\,dp $$ 
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > stiffness( const Indexer &indexer ) const;


	// Computes the stiffness matrix w.r.t. to the piecewise-constant inner-product given by the matrices
	//tex:
	// $$S(x,y) = \sum_{e\in E}\int_e\langle\nabla x|_p,w[e] \cdot \nabla y|_p\rangle\,dp = \sum_{e\in E}\int_e\langle w[e],\nabla x|_p\otimes \nabla y|_p\rangle\,dp$$
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename InnerProductFunctor /* = std::function< MishaK::SquareMatrix< double , Dim > ( Index< Dim > E/F [ , unsigned int t ] ) > */ >
	Eigen::SparseMatrix< double > stiffness( const Indexer & indexer , InnerProductFunctor && ipf , bool piecewiseLinear=false ) const;

	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
	Eigen::SparseMatrix< double > deltaMass( const Indexer & indexer , SampleFunctor && F , size_t sampleNum , WeightFunctor && wF ) const;
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > , unsigned int ) */ >
	Eigen::SparseMatrix< double > deltaMass( const Indexer & indexer , SampleFunctor && F , const OrderedSampler< Dim > &orderedSampler , WeightFunctor && wF ) const;
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
	Eigen::SparseMatrix< double > deltaMass( const Indexer & indexer , const std::vector< Point< double , Dim > > &samples , WeightFunctor && wF ) const;
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ >
	Eigen::SparseMatrix< double > deltaMass( const Indexer & indexer , SampleFunctor && F , size_t sampleNum ) const;
	template< typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ >
	Eigen::SparseMatrix< double > deltaMass( const Indexer & indexer , SampleFunctor && F , const OrderedSampler< Dim > &orderedSampler ) const;
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > deltaMass( const Indexer & indexer , const std::vector< Point< double , Dim > > &samples ) const;

	// Returns the matrix giving the mass along the boundary
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > boundaryMass( const Indexer &indexer ) const;

	// Returns the matrix giving the stiffness along the boundary
	template< typename Indexer /* = Hat::Indexer< Dim > */ >
	Eigen::SparseMatrix< double > boundaryStiffness( const Indexer &indexer ) const;

	// Returns the prolgonation matrix from the current level to the next finer one
	template< typename ProlongationIndexer /* = Hat::BaseProlongationIndexer< Dim > */ >
	Eigen::SparseMatrix< double > prolongation( const ProlongationIndexer &pIndexer , size_t numFineFunctions ) const;

	// Returns the prolgonation of a vector from the current level to the next finer one
	template< typename ProlongationIndexer /* = Hat::BaseProlongationIndexer< Dim > */ >
	Eigen::VectorXd prolongation( const Eigen::VectorXd &x , const ProlongationIndexer &pIndexer , size_t numFineFunctions ) const;
	
protected:

	unsigned int _r;
	Range< Dim > _fRange , _eRange;
};
#include "Hat.scalars.inl"
