#include "Include/PreProcessor.h"

#include <vector>
#include <unordered_map>
#include <sstream>
#include <filesystem>
#include <Eigen/Eigenvalues>
#include "Misha/Miscellany.h"
#include "Misha/Geometry.h"
#include "Misha/RegularGrid.h"
#include "Misha/Algebra.h"
#include "Include/GridSamples.h"
#include "SymmetricPoissonRecon.Solvers.h"

namespace MishaK
{
	namespace SymPR
	{
		// Note that the functions here take values in the unit cube whereas regular grids take values in the range defined by the resolution

		// Computes the transformation mapping the point set into the unit cube
		template< unsigned int Dim >
		SquareMatrix< double, Dim + 1 > ToUnitCube( std::function< Point< double, Dim >( size_t idx ) > F, size_t sampleNum, double scale);

		// Computes a regular grid of splatted values
		template< bool FirstOrder , typename FieldData , unsigned int Dim >
		static void SetField
		(
			GridSamples::TreeSplatter< Dim , FieldData > &field ,
			unsigned int depth,														// The depth of the grid
			std::function< std::pair< Point< double, Dim >, FieldData >( size_t ) > F,	// The samples to be splatted in
			const OrderedSampler< Dim > &orderedSampler ,
			const GridSamples::Estimator< Dim >& estimator ,						// A measure of the sampling density (and possibly noise)
			std::function< double ( unsigned int ) > splatScale,					// The factor by which all samples splatted in, as a function of depth
			unsigned int kernelRadius,												// The radius of a splat
			unsigned int verbosity													// The verbosity of the output
		);

		template< unsigned int Dim, bool Sym, typename Energy >
		struct Reconstructor
		{
			using SolutionType = std::conditional_t< Sym, Eigen::VectorXd, std::pair< Eigen::VectorXd, Eigen::VectorXd > >;
			using SampleData = Hat::SquareMatrix< double, Dim, Sym >;

			template< typename HierarchicalIndexer >
			Reconstructor
			(
				const HierarchicalIndexer & hierarchicalIndexer ,
				unsigned int depth , unsigned int minSolveDepth ,
				std::function< std::pair< Point< double , Dim > , SampleData >( size_t ) > F , size_t sampleNum ,
				const OrderedSampler< Dim >& orderedSampler ,
				const GridSamples::Estimator< Dim >& estimator ,
				int adaptiveDilationRadius ,
				bool firstOrder ,
				double boundaryStiffnessWeight ,
				double screeningWeight ,
				bool depthWeightedScreening ,
				unsigned int kernelRadius ,
				unsigned int verbosity
			);

			//SolutionType solve
			template< typename HierarchicalIndexer >
			std::vector< SolutionType > solve
			(
				unsigned int solveDepth ,
				const HierarchicalIndexer & hierarchicalIndexer ,
				const SolutionType& initialGuess, bool rescale,
				bool singleLevel,
				double iterationMultiplier,
				unsigned int coarseIterations,
				unsigned int iterations,
#ifdef USE_NL_OPT
				unsigned int gradientMemory,
#endif // USE_NL_OPT
				PoissonSolvers::UpdateType coarseSolver,
				PoissonSolvers::UpdateType fineSolver,
				unsigned int verbosity
			);
		protected:
			template< typename Indexer >
			static Eigen::SparseMatrix< double > _GetScreeningRegularizer
			(
				const Indexer & indexer ,
				unsigned int depth ,
				std::function< Point< double, Dim > ( size_t ) > F ,
				const OrderedSampler< Dim > &orderedSampler ,
				const GridSamples::Estimator< Dim >& estimator ,
				double screeningWeight ,
				bool depthWeightedScreening ,
				bool verbose
			);

			template< bool FirstOrder , typename Indexer >
			static Energy _GetEnergy
			(
				const Indexer & indexer ,
				unsigned int depth,
				std::function< std::pair< Point< double, Dim > , SampleData > ( size_t ) > F , size_t sampleNum ,
				const OrderedSampler< Dim > &orderedSampler ,
				const GridSamples::Estimator< Dim >& estimator ,
				double screeningWeight ,
				bool depthWeightedScreening ,
				unsigned int kernelRadius ,
				unsigned int verbosity
			);

			unsigned int _depth, _minSolveDepth;
			Hat::Range< Dim > _fRange;
			std::vector< Energy > _energies;
			std::vector< SolutionType > _solution;
			std::vector< std::vector< std::vector< unsigned int > > > _mcIndices;
		};

		////////////////////
		// Implementation //
		////////////////////
		template< unsigned int Dim >
		SquareMatrix< double, Dim + 1 > ToUnitCube( std::function< Point< double , Dim > (size_t idx ) > F , size_t sampleNum , double scale )
		{
			if (!sampleNum) MK_ERROR_OUT("Expected some samples: ", sampleNum);

			SquareMatrix< double, Dim + 1 > xForm = SquareMatrix< double, Dim + 1 >::Identity();
			Point< double, Dim > bBox[2];
			bBox[0] = bBox[1] = F(0);
			for (unsigned int i = 1; i < sampleNum; i++) for (unsigned int d = 0; d < Dim; d++)
			{
				Point< double, Dim > p = F(i);
				bBox[0][d] = std::min< double >(bBox[0][d], p[d]);
				bBox[1][d] = std::max< double >(bBox[1][d], p[d]);
			}
			double _scale = bBox[1][0] - bBox[0][0];
			for (unsigned int d = 1; d < Dim; d++) _scale = std::max< double >(_scale, bBox[1][d] - bBox[0][d]);

			SquareMatrix< double, Dim + 1 > t1 = SquareMatrix< double, Dim + 1 >::Identity(), s = SquareMatrix< double, Dim + 1 >::Identity(), t2 = SquareMatrix< double, Dim + 1 >::Identity();
			for (unsigned int d = 0; d < Dim; d++) t2(Dim, d) = -(bBox[0][d] + bBox[1][d]) / 2.;
			for (unsigned int d = 0; d < Dim; d++) s(d, d) = 1. / (_scale * scale);
			for (unsigned int d = 0; d < Dim; d++) t1(Dim, d) = 0.5;
			xForm = t1 * s * t2;
			return xForm;
		}


		template< bool FirstOrder , typename FieldData, unsigned int Dim >
		static void SetField
		(
			GridSamples::TreeSplatter< Dim , FieldData > &field ,
			unsigned int depth ,
			std::function< std::pair< Point< double , Dim > , FieldData > ( size_t ) > F ,
			const OrderedSampler< Dim > &orderedSampler ,
			const GridSamples::Estimator< Dim >& estimator ,
			std::function< double ( unsigned int ) > splatScale ,
			unsigned int kernelRadius ,
			unsigned int verbosity
		)
		{
			{
				// [WARNING] This uses the finest depth
				auto _F = [&]( size_t idx )
					{
						std::pair< Point< double, Dim >, FieldData > sample = F(idx);

						// Have the magnitude of the sample's contribution be proportional to its measure [i.e. number of 
						sample.second *= estimator.measure( sample.first , 0 );
						return sample;
					};

				switch( kernelRadius )
				{
				case 0: field.template addSamples< 0 >( _F , orderedSampler , [&]( size_t idx ){ return estimator.depth( F(idx).first , 0 ); } , splatScale , false ) ; break;
				case 1: field.template addSamples< 1 >( _F , orderedSampler , [&]( size_t idx ){ return estimator.depth( F(idx).first , 0 ); } , splatScale , false ) ; break;
				case 2: field.template addSamples< 2 >( _F , orderedSampler , [&]( size_t idx ){ return estimator.depth( F(idx).first , 0 ); } , splatScale , false ) ; break;
				case 3: field.template addSamples< 3 >( _F , orderedSampler , [&]( size_t idx ){ return estimator.depth( F(idx).first , 0 ); } , splatScale , false ) ; break;
				case 4: field.template addSamples< 4 >( _F , orderedSampler , [&]( size_t idx ){ return estimator.depth( F(idx).first , 0 ); } , splatScale , false ) ; break;
				default: MK_ERROR_OUT( "Kernel radius must be in range [0,4]: " , kernelRadius );
				}
			}
		}

		///////////////////
		// Reconstructor //
		///////////////////
		template< unsigned int Dim , bool Sym , typename Energy >
		template< typename Indexer >
		Eigen::SparseMatrix< double > Reconstructor< Dim , Sym , Energy >::_GetScreeningRegularizer
		(
			const Indexer & indexer ,
			unsigned int depth ,
			std::function< Point< double , Dim > ( size_t ) > F ,
			const OrderedSampler< Dim > &orderedSampler ,
			const GridSamples::Estimator< Dim > &estimator ,
			double screeningWeight ,
			bool depthWeightedScreening ,
			bool verbose
		)
		{
			Hat::ScalarFunctions< Dim > scalars( 1<<depth );
			Eigen::SparseMatrix< double > R( indexer.numFunctions() , indexer.numFunctions() );

			double measure = 0;
			ThreadPool::ParallelFor( 0 , orderedSampler.numSamples() , [&]( unsigned int thread , size_t i ){ Atomic< double >::Add( measure , estimator.measure( F(i) , thread ) ); } );
			if( screeningWeight>0 )
			{
				if( depthWeightedScreening )
				{
					auto weightFunctor = [&]( Point< double , Dim > p , unsigned int thread ) -> double
						{
							double d = estimator.depth( p , thread ) - depth + 1;
							d = std::min< double >( std::max< double >( d , 0 ) , 1 );
							return d;
						};
					R = scalars.deltaMass( indexer , [&]( size_t idx ){ return F(idx); } , orderedSampler , weightFunctor ) * screeningWeight * measure / orderedSampler.numSamples();
				}
				else R = scalars.deltaMass( indexer , [&]( size_t idx ){ return F(idx); } , orderedSampler , [&]( Point< double , Dim > p , unsigned int thread ){ return 1.; } ) * screeningWeight * measure / orderedSampler.numSamples();
			}
			return R;
		}

		template< unsigned int Dim , bool Sym , typename Energy >
		template< bool FirstOrder , typename Indexer >
		Energy Reconstructor< Dim , Sym , Energy >::_GetEnergy
		(
			const Indexer & indexer ,
			unsigned int depth ,
			std::function< std::pair< Point< double , Dim > , SampleData > ( size_t ) > F , size_t sampleNum ,
			const OrderedSampler< Dim > &orderedSampler ,
			const GridSamples::Estimator< Dim > &estimator ,
			double screeningWeight ,
			bool depthWeightedScreening , 
			unsigned int kernelRadius ,
			unsigned int verbosity
		)
		{
			Miscellany::PerformanceMeter pMeter( '.' );

			Eigen::SparseMatrix< double > R;
			R = _GetScreeningRegularizer( indexer , depth , [&]( size_t idx ){ return F(idx).first; } , orderedSampler , estimator , screeningWeight , depthWeightedScreening , verbosity>=2 );

			if( verbosity ) std::cout << pMeter( "Set regularizer" ) << std::endl;

			Hat::ScalarFunctions< Dim > scalars( 1 << depth );

			// Assume that we are given a d-dimensional manifold M living in \R^n, with a map x:M -> \otimes^r\R^n
			// We define a product field on \R^n via a unit-norm L_1 kernel K at level l as:
			//		X_l(p) = \int_M K_l(p-q) * x_l(q) dq
			// The square norm of X_l is:
			//		|| X_l ||^2 = \int_{\R^n} ( \int_M K_l(p-q) * x_l(q) ) dq )^2 dp
			// Transitioning to the next level effects things in two ways:
			//		1. x_{l+1}    <- 2^r * x_l:			to account for the fact that we want the gradients to be twice as large
			//		2. K_{l+1}(p) <- 2^n * K_l(p*2):	to account for the fact that the kernel gets twice as narrow but is still unit-norm
			// This gives:
			//		X_{l+1}(p) = \int_M K_{l+1}( (p-q)*2 ) * x_{l+1}(q) dq
			//		           = 2^{n+r} * \int_M K_l( (p-q)*2 ) * x_l(q) dq
			// Assuming, for simplicity, that x_l(q) is constant
			//		           ~ 2^{n+r} * x_l * \int_M K_l( (p-q)*2 ) dq
			// Then the square norm is:
			//		|| X_{l+1} ||^2 ~ 2^{2n+2r} * || x_l ||^2 * \int_{\R^n} ( \int_M K_l( (p-q)*2 ) dq )^2 dp
			// Assuming, further, that K_l is the indicator function of a unit ball gives:
			//		                ~ 2^{2n+2r} * || x_l ||^2 * \int_{\R^n} ( |B_0.5^n(p) \cap M| )^2 dp
			//		                ~ 2^{2n+2r} * || x_l ||^2 * \int_{ p | d(p,M)<0.5 } |B_0.5^n(p) \cap M|^2 dp
			// Assuming that M is flat, we have:
			//		|B_0.5^n(p) \cap M| = |B_{sqrt( (0.5)^2 - d^2(p,M) )}}^d|
			//		                    ~ 2^{-d} 
			// This gives:
			//		|| X_{l+1} ||^2 ~ 2^{2n+2r} * || x_l ||^2 * \int_{ p | d(p,M)<0.5 } (2^{-d})^2
			//		                = 2^{2n+2r} * || x_l ||^2 * 2^{-2d} * \int_{ p |d(p,M)<0.5 }
			//		                ~ 2^{2n+2r-2d} * || x_l ||^2 * |M| * 2^{d-n}
			//		                ~ 2^{2r+n-d} * || x_l ||^2 * |M|
			// So that:
			//		|| X_{l+1} || ~ sqrt( 2^{2r+n-d} )
			//                    = 2^{r + 0.5(n-d)}

			std::function< double ( unsigned int ) > splatScale = []( unsigned int depth ){ return pow( 2. , depth*Dim + depth*( 2. - 1.5 ) ); };

			GridSamples::TreeSplatter< Dim , SampleData > field( depth );
			SetField< FirstOrder , SampleData >( field , depth , F , orderedSampler , estimator , splatScale , kernelRadius , verbosity );
			if( verbosity ) std::cout << pMeter( "Splatted" ) << std::endl;

			pMeter.reset();
			Energy e( indexer , 1<<depth , field , FirstOrder , 0 , R );
			if( verbosity ) std::cout << pMeter( "Got energy" ) << std::endl;
			return e;
		}

		template< unsigned int Dim , bool Sym , typename Energy >
		template< typename HierarchicalIndexer >
		Reconstructor< Dim, Sym, Energy >::Reconstructor
		(
			const HierarchicalIndexer & hierarchicalIndexer ,
			unsigned int depth , unsigned int minSolveDepth ,
			std::function< std::pair< Point< double , Dim > , SampleData > ( size_t ) > F , size_t sampleNum ,
			const OrderedSampler< Dim >& orderedSampler ,
			const GridSamples::Estimator< Dim > &estimator ,
			int adaptiveDilationRadius ,
			bool firstOrder ,
			double boundaryStiffnessWeight ,
			double screeningWeight ,
			bool depthWeightedScreening ,
			unsigned int kernelRadius ,
			unsigned int verbosity
		)
			: _depth(depth) , _minSolveDepth(minSolveDepth)
		{
			using Indexer = typename HierarchicalIndexer::Indexer;
			static_assert( std::is_base_of_v< Hat::BaseHierarchicalIndexer< Dim > , HierarchicalIndexer > , "[ERROR] Poorly formed HierarchicalIndexer" );
			static_assert( std::is_base_of< ProductSystem::Energy< Dim , true , Indexer > , Energy >::value , "[ERROR] Expected ProductSystem::Energy< Dim , true >" );
			Miscellany::PerformanceMeter pMeter( '.' );

			Hat::ScalarFunctions< Dim > scalars( 1<<_depth );
			for( unsigned int d=0 ; d<Dim ; d++ ) _fRange.first[d] = 0 , _fRange.second[d] = (1<<_depth) + 1;

			_solution.resize( _depth+1 );

			{
				Miscellany::PerformanceMeter pMeter( '.' );
				_mcIndices.resize( _depth + 1 );

				for( unsigned int d=0 ; d<=_depth ; d++ )
				{
					auto indexer = hierarchicalIndexer[d];
					Hat::ScalarFunctions< Dim > scalars( 1<<d );

					_solution[d].resize( indexer.numFunctions() );

					_mcIndices[d].resize( 1<<Dim );

					for( unsigned int f=0 ; f<indexer.numActiveFunctions() ; f++ )
					{
						Hat::Index< Dim > F = indexer.functionIndex( f );
						unsigned mcIndex = 0;
						for( unsigned int d=0 ; d<Dim ; d++ ) if( F[d]&1 ) mcIndex |= (1<<d);
						_mcIndices[d][ mcIndex ].push_back( f );
					}
					if( verbosity>=3 ) std::cout << "DoFs[ " << d << " ] " << indexer.numActiveFunctions() << " / " << scalars.functionNum() << std::endl;
				}

				if( verbosity ) std::cout << pMeter( "MC indices" ) << std::endl;
			}

			{
				_energies.reserve( _depth+1 );
				if( firstOrder ) _energies.emplace_back( _GetEnergy< true  >( hierarchicalIndexer[depth] , depth , F , sampleNum , orderedSampler , estimator , screeningWeight , depthWeightedScreening , kernelRadius , verbosity ) );
				else             _energies.emplace_back( _GetEnergy< false >( hierarchicalIndexer[depth] , depth , F , sampleNum , orderedSampler , estimator , screeningWeight , depthWeightedScreening , kernelRadius , verbosity ) );
				{
					Miscellany::PerformanceMeter pMeter('.');
					for( unsigned int depth=0 ; depth<_depth ; depth++ ) _energies.emplace_back( _energies.back().restrict( hierarchicalIndexer[_depth-1-depth] ) );
					if( verbosity ) std::cout << pMeter( "Restricted" ) << std::endl;
				}
			}

			if( depthWeightedScreening )
			{
				Miscellany::PerformanceMeter pMeter( '.' );
				for( int depth=(int)_depth-1 ; depth>=(int)_minSolveDepth ; depth-- ) _energies[_depth-depth].regularizer() = _GetScreeningRegularizer( hierarchicalIndexer[depth] , depth , [&]( size_t idx ){ return F(idx).first; } , orderedSampler , estimator , screeningWeight , depthWeightedScreening , verbosity>=2 );
				if( verbosity ) std::cout << pMeter( "Got regularizers" ) << std::endl;
			}
			if( boundaryStiffnessWeight>0 )
			{
				Miscellany::PerformanceMeter pMeter('.');
				for( unsigned int d=0 ; d<=_depth ; d++ )
				{
					Hat::ScalarFunctions< Dim > scalars( 1<<d );
					_energies[_depth-d].regularizer() += scalars.boundaryStiffness( hierarchicalIndexer[d] ) * boundaryStiffnessWeight;
				}
				if( verbosity ) std::cout << pMeter( "Boundary stiffness" ) << std::endl;
			}

			if( verbosity ) std::cout << pMeter( "Set constraints" ) << std::endl;
		}

		template< unsigned int Dim , bool Sym , typename Energy >
		template< typename HierarchicalIndexer >
		std::vector<typename Reconstructor< Dim , Sym , Energy >::SolutionType> Reconstructor< Dim , Sym , Energy >::solve
		(
			unsigned int solveDepth ,
			const HierarchicalIndexer & hierarchicalIndexer ,
			const SolutionType &initialGuess , bool rescale ,
			bool singleLevel ,
			double iterationMultiplier ,
			unsigned int coarseIterations ,
			unsigned int iterations ,
#ifdef USE_NL_OPT
			unsigned int gradientMemory ,
#endif // USE_NL_OPT
			PoissonSolvers::UpdateType coarseSolver ,
			PoissonSolvers::UpdateType fineSolver ,
			unsigned int verbosity
		)
		{
			solveDepth = std::max< unsigned int >( _minSolveDepth , std::min< unsigned int >( _depth , solveDepth ) );

			Miscellany::PerformanceMeter pMeter( '.' );

			auto Iters = [&]( unsigned int iters , unsigned int depth ){ return (int)ceil( pow( iterationMultiplier , _depth - depth ) * iters ); };

			auto MCIndices = [&]( unsigned int d ) -> const std::vector< std::vector< unsigned int > > & { return _mcIndices[d]; };


			auto X = [&]( unsigned int d ) -> Eigen::VectorXd&
				{
					if constexpr( Sym ) return _solution[d];
					else                return _solution[d].first;

				};
			auto Y = [&]( unsigned int d ) -> Eigen::VectorXd&
				{
					if constexpr( Sym ) return _solution[d];
					else                return _solution[d].second;
				};
			auto ClearSolution = [&]( unsigned int d )
				{
					if constexpr( Sym ) _solution[d].setZero();
					else _solution[d].first.setZero(), _solution[d].second.setZero();
				};
			auto SetProlongedSolution = [&]( unsigned int d )
				{
					if constexpr( Sym ) _solution[d+1] = _energies[_depth-d].scalarProlongation( _solution[d] );
					else
					{
						_solution[d+1].first  = _energies[_depth-d].scalarProlongation( _solution[d].first  );
						_solution[d+1].second = _energies[_depth-d].scalarProlongation( _solution[d].second );
					}
				};
			auto AddProlongedSolution = [&]( unsigned int d )
				{
					if constexpr( Sym ) _solution[d+1] += _energies[_depth-d].scalarProlongation( _solution[d] );
					else
					{
						_solution[d+1].first += _energies[_depth-d].scalarProlongation( _solution[d].first );
						_solution[d+1].second += _energies[_depth-d].scalarProlongation( _solution[d].second );
					}
				};
			auto RescaleSolution = [&]( unsigned int d , bool full )
				{
					if( full )
					{
						Polynomial::Polynomial1D< 4 > q;
						if constexpr( Sym ) q = _energies[_depth-d].scalingQuarticFit( _solution[d] );
						else                q = _energies[_depth-d].scalingQuarticFit( _solution[d].first , _solution[d].second );
						double roots[3];
						Polynomial::Polynomial1D< 3 > dQ = q.d( 0 );
						unsigned int rootCount = Polynomial::Roots( dQ , roots );
						if( !rootCount ) MK_ERROR_OUT( "Expected roots" );
						double s = roots[0];
						for( unsigned int d=1 ; d<rootCount ; d++ ) if( q(roots[d]) < q(s) ) s = roots[d];
						if constexpr( Sym ) _solution[d] *= s;
						else                _solution[d].first *= s , _solution[d].second *=s;
					}
					else
					{
						const Hat::ScalarFunctions< Dim > &scalars = _energies[_depth-d].scalars;
						const Hat::ProductFunctions< Dim , Sym > &products = _energies[_depth-d].products;
						typename Hat::ProductFunctions< Dim , Sym >::template IntegrationStencil< double , 2 , 0 > stencil = Hat::ProductFunctions< Dim , Sym >::MassStencil( scalars.resolution() );
						typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double > fullStencil( stencil , scalars.resolution() );
						double dot;
						if constexpr( Sym )
						{
							dot = products( hierarchicalIndexer[d] , fullStencil , _solution[d] , _solution[d] , _solution[d] , _solution[d] );
						}
						else dot = products( fullStencil , _solution[d].first , _solution[d].second , _solution[d].first , _solution[d].second );
						double s = pow( _energies[_depth-d].targetSquareNorm() / dot , 0.25 );
						if constexpr( Sym ) _solution[d] *= s;
						else _solution[d].first *= s , _solution[d].second *= s;
					}
				};

			{
				_solution[_minSolveDepth] = initialGuess;
				if( rescale ) RescaleSolution( _minSolveDepth , false );
//				if( rescale ) RescaleSolution( _minSolveDepth , true );

				if( singleLevel )
				{
					// Prolong from coarsest to finest and clear all the coarser levels
					for( unsigned int d=_minSolveDepth ; d<solveDepth ; d++ ) SetProlongedSolution( d );
				}
			}
			// Clear all levels except for the initial guess
			if( singleLevel ) for( unsigned int d=_minSolveDepth   ; d< _depth ; d++ ){ if( d!=solveDepth ) ClearSolution( d ); }
			else              for( unsigned int d=_minSolveDepth+1 ; d<=_depth ; d++ ) ClearSolution( d );

			double _err = _energies[0].targetSquareNorm();


			if( verbosity ) std::cout << pMeter( "Init solution" ) << std::endl;

			if( singleLevel )
			{
				unsigned int d = solveDepth;
				PoissonSolvers::ParallelGaussSeidelSolve< Dim >( d , MCIndices( d ) , _energies[_depth-d] , _solution[d] , iterations , verbosity );
			}
			else
			{
				double err = 0;
				if( verbosity>=1 )
					if( X(solveDepth).isZero() && Y(solveDepth).isZero() ) err = _energies[0].targetSquareNorm();
					else err = _energies[0]( X( solveDepth ) , Y( solveDepth ) );
				{

					// Fine-to-coarse
					for( unsigned int d=solveDepth ; d+1!=_minSolveDepth ; d-- )
					{
						if( d>_minSolveDepth ) ClearSolution( d );
						// Restrict to the next resolution
						if( d>_minSolveDepth ) _energies[_depth-(d-1)].update( _energies[_depth-d] , X(d) , Y(d) );
					}

					if( verbosity ) std::cout << pMeter( "Updated" ) << std::endl;

					// Coarse-to-fine
					for( unsigned int d=_minSolveDepth ; d<=solveDepth ; d++ )
					{
						//					MK_WARN_ONCE( "Rescaling next solution" );
						//					RescaleSolution( d ,true );
						//					RescaleSolution( d ,false );
						// Solve at the current resolution


						if( d==_minSolveDepth && coarseIterations ) // at coarser level we don't do adaptive because we need the full grid
						{
							switch( coarseSolver )
							{
							case PoissonSolvers::UpdateType::SERIAL_GAUSS_SEIDEL:
								PoissonSolvers::SerialGaussSeidelSolve< Dim >( d , _energies[_depth-d] , _solution[d] , coarseIterations , verbosity );
								break; 
							case PoissonSolvers::UpdateType::PARALLEL_GAUSS_SEIDEL:
								PoissonSolvers::ParallelGaussSeidelSolve< Dim >( d , MCIndices( d ) , _energies[_depth-d] , _solution[d] , coarseIterations , verbosity );
								break;
							case PoissonSolvers::UpdateType::GRADIENT_DESCENT:
								PoissonSolvers::    GradientDescentSolve< Dim >( d ,                  _energies[_depth-d] , _solution[d] , coarseIterations , verbosity );
								break;
							case PoissonSolvers::UpdateType::NEWTON:
								PoissonSolvers::             NewtonSolve< Dim >( d ,                  _energies[_depth-d] , _solution[d] , coarseIterations , verbosity );
								break;
#ifdef USE_NL_OPT
							case PoissonSolvers::UpdateType::NL_OPT_MMA:
								PoissonSolvers::NLOptSolve< Dim >( nlopt::algorithm::LD_MMA                     , d , _energies[_depth-d] , _solution[d] , coarseIterations , gradientMemory , verbosity );
								break;
							case PoissonSolvers::UpdateType::NL_OPT_LBFGS:
								PoissonSolvers::NLOptSolve< Dim >( nlopt::algorithm::LD_LBFGS                   , d , _energies[_depth-d] , _solution[d] , coarseIterations , gradientMemory , verbosity );
								break;
							case PoissonSolvers::UpdateType::NL_TNEWTON_PRECOND_RESTART:
								PoissonSolvers::NLOptSolve< Dim >( nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART , d , _energies[_depth-d] , _solution[d] , coarseIterations , gradientMemory , verbosity );
								break;
#endif // USE_NL_OPT
							default: 
								MK_ERROR_OUT( "Unrecognized coarse solver type" );
							}
						}
						else
						{
							switch( fineSolver )
							{
							case PoissonSolvers::UpdateType::SERIAL_GAUSS_SEIDEL:
								PoissonSolvers::  SerialGaussSeidelSolve< Dim >( d ,                  _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , verbosity );
								break;
							case PoissonSolvers::UpdateType::PARALLEL_GAUSS_SEIDEL:
								PoissonSolvers::ParallelGaussSeidelSolve< Dim >( d , MCIndices( d ) , _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , verbosity );
								break;
							case PoissonSolvers::UpdateType::GRADIENT_DESCENT:
								PoissonSolvers::    GradientDescentSolve< Dim >( d ,                  _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , verbosity );
								break;
							case PoissonSolvers::UpdateType::NEWTON:
								PoissonSolvers::             NewtonSolve< Dim >( d ,                  _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , verbosity );
								break;
#ifdef USE_NL_OPT
							case PoissonSolvers::UpdateType::NL_OPT_MMA:
								PoissonSolvers::NLOptSolve< Dim >( nlopt::algorithm::LD_MMA                     , d , _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , gradientMemory , verbosity );
								break;
							case PoissonSolvers::UpdateType::NL_OPT_LBFGS:
								PoissonSolvers::NLOptSolve< Dim >( nlopt::algorithm::LD_LBFGS                   , d , _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , gradientMemory , verbosity );
								break;
							case PoissonSolvers::UpdateType::NL_TNEWTON_PRECOND_RESTART:
								PoissonSolvers::NLOptSolve< Dim >( nlopt::algorithm::LD_TNEWTON_PRECOND_RESTART , d , _energies[_depth-d] , _solution[d] , Iters( iterations , d ) , gradientMemory , verbosity );
								break;
#endif // USE_NL_OPT
							default: 
								MK_ERROR_OUT( "Unrecognized coarse solver type" );
							}
						}

						// Prolong to the next resolution
						if( d<solveDepth ) AddProlongedSolution(d);
					}
				}
			}
			if( verbosity )
			{
				std::pair< double , double > errs = _energies[_depth-solveDepth].energies( X( solveDepth ) , Y( solveDepth ) );
				std::cout << "Error[" << _energies[_depth-solveDepth].scalars.resolution() << "]: " << _err << " -> " << errs.first + errs.second << " = " << errs.first << " + " << errs.second << " => " << sqrt( (errs.first+errs.second )/_err ) << std::endl;
			}

			// Remove the part from the higher frequency that was prolonged from the lower frequency
			for( unsigned int d=solveDepth ; d>_minSolveDepth ; d-- ) _solution[d] -= _energies[_depth-(d-1)].scalarProlongation( _solution[d-1] );
			for( unsigned int d=0 ; d<_minSolveDepth ; d++ ) _solution[d].setZero();

			return _solution;
		}
	}
}