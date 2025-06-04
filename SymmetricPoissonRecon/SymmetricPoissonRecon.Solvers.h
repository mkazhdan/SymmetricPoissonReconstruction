#ifndef POISSON_SOLVERS_INCLUDED
#define POISSON_SOLVERS_INCLUDED


#include <vector>
#include <iomanip>
#include <Eigen/Sparse>
#include "Misha/Geometry.h"
#include "Misha/Miscellany.h"
#include "Misha/Poly34.h"
#include "Misha/Polynomial.h"
#include "Include/Hat.h"
#include "Include/MG.h"
#include "Include/SystemEnergy.Products.h"
//#include "boost/align.hpp"
#ifdef USE_NL_OPT
//#include "nlopt.hpp"
#include "nlopt/nlopt.hpp"
#endif // USE_NL_OPT


namespace MishaK
{
	namespace PoissonSolvers
	{
		enum UpdateType
		{
			SERIAL_GAUSS_SEIDEL,
			PARALLEL_GAUSS_SEIDEL,
			GRADIENT_DESCENT,
#ifdef USE_NL_OPT
			NEWTON,
			NL_OPT_MMA,
			NL_OPT_LBFGS,
			NL_TNEWTON_PRECOND_RESTART,
#else // !USE_NL_OPT
			NEWTON
#endif // USE_NL_OPT
		};
#ifdef USE_NL_OPT
		static const unsigned int UpdateTypeCount = 5;
		static const std::string UpdateTypeNames[] =
		{
			std::string("serial Gauss-Seidel") ,
			std::string("parallel Gauss-Seidel") ,
			std::string("gradient descent") ,
			std::string("Newton") ,
			std::string("non-linear optimization (method of moving asymptotes)") ,
			std::string("non-linear optimization (limited-memory BFGS)") ,
			std::string("non-linear optimization (preconditioned truncated Newton with restarting)")
		};
#else // !USE_NL_OPT
		static const unsigned int UpdateTypeCount = 4;
		static const std::string UpdateTypeNames[] = { std::string("serial Gauss-Seidel") , std::string("parallel Gauss-Seidel") , std::string("gradient descent") , std::string("Newton") };
#endif // USE_NL_OPT


		// Given the current estimate of the solution x, updates the estimate of the solution by performing Gauss-Seidel relaxation one coefficient at a time
		template< unsigned int Dim, typename SystemEnergyType >
		void SerialGaussSeidelSolve(unsigned int depth, const SystemEnergyType& energy, Eigen::VectorXd& x, unsigned int iters, int verbosity);

		// Given the current estimate of the solution x, updates the estimate of the solution by performing Gauss-Seidel relaxation one coefficient at a time
		template< unsigned int Dim, typename SystemEnergyType >
		void ParallelGaussSeidelSolve(unsigned int depth, const std::vector< std::vector< unsigned int > >& mcIndices, const SystemEnergyType& energy, Eigen::VectorXd& x, unsigned int iters, int verbosity, bool adaptive=false );

		// Given the current estimate of the solution x, updates the estimate of the solution by performing Gauss-Seidel relaxation one coefficient at a time
		template< unsigned int Dim, typename SystemEnergyType >
		void GradientDescentSolve(unsigned int depth, const SystemEnergyType& energy, Eigen::VectorXd& x, unsigned int iters, int verbosity);

		// Given the current estimate of the solution x, updates the estimate of the solution by performing Gauss-Seidel relaxation one coefficient at a time
		template< unsigned int Dim, typename SystemEnergyType >
		void NewtonSolve(unsigned int depth, const SystemEnergyType& energy, Eigen::VectorXd& x, unsigned int iters, int verbosity);

#ifdef USE_NL_OPT
		template< unsigned int Dim, typename SystemEnergyType >
		void NLOptSolve(nlopt::algorithm algoType, unsigned int depth, const SystemEnergyType& energy, Eigen::VectorXd& x, unsigned int iters, unsigned int gradientMemory, int verbosity);
#endif // USE_NL_OPT

		/////////////
		// Solvers //
		/////////////

		template< unsigned int Dim, typename SystemEnergyType >
		void SerialGaussSeidelSolve(unsigned int depth, const SystemEnergyType& energy, Eigen::VectorXd& x, unsigned int iters, int verbosity)
		{
			static const bool Sym = true;

			static_assert( std::is_base_of< ProductSystem::Energy< Dim , Sym , typename SystemEnergyType::Indexer > , SystemEnergyType >::value , "[ERROR] SystemEnergyType poorly formed" );
			static const double EPS = 1e-24;
			Miscellany::Timer timer;
			Hat::ScalarFunctions< Dim > scalarFunctions(1 << depth);

			double err = 0.;
			if( verbosity>=2 ) err = energy(x);
			for( unsigned int iter=0 ; iter<iters ; iter++ )
			{
				for( unsigned int idx=0 ; idx<energy.indexer.numFunctions() ; idx++ )
				{
					Polynomial::Polynomial1D< 4 > _Q = energy.quarticFit( x , idx , 0 );
					for( unsigned int i=0 ; i<=4 ; i++ ) if( fabs( _Q.coefficient(i) )<EPS ) _Q.coefficient(i) = 0;
					Polynomial::Polynomial1D< 3 > _dQ = _Q.d(0);
					double roots[3];
					unsigned int cnt = Polynomial::Roots( _dQ , roots , EPS );
					if( !cnt ) MK_ERROR_OUT( "Expected roots!" );
					double s = roots[0];
					for( unsigned int i=1 ; i<cnt ; i++ ) if( s!=s || _Q(roots[i])<_Q(s) ) s = roots[i];
					if( s!=s ) s=0;
					x[idx] += s;
				}
			}
			if( verbosity>=2 )
			{
				std::pair< double, double > _err = energy.energies(x);
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 5 );
					std::cout << "\tError[" << std::setw(3) << (1 << depth) << "]: " << std::setw(9) << err << " -> " << (_err.first + _err.second) << " = " << _err.first << " + " << _err.second;
					std::cout << "\titers = " << iters;
				}
				{
					Miscellany::StreamFloatPrecision sfp(std::cout, 2);
					std::cout << "\t" << std::setw(6) << timer.elapsed() << " (s)" << std::endl;
				}
			}
		}


		template< unsigned int Dim , typename SystemEnergyType >
		void ParallelGaussSeidelSolve( unsigned int depth , const std::vector< std::vector< unsigned int > > &mcIndices , const SystemEnergyType &energy , Eigen::VectorXd &x , unsigned int iters , int verbosity, bool adaptive )
		{
			static const bool Sym = true;
			static_assert( std::is_base_of< ProductSystem::Energy< Dim , Sym , typename SystemEnergyType::Indexer > , SystemEnergyType >::value , "[ERROR] SystemEnergyType poorly formed" );
			static const double EPS = 1e-24;
			Miscellany::Timer timer;

			double err = 0.;
			if( verbosity>=2 )
			{
				if( !adaptive ) err = energy(x);
			}
			for( unsigned int iter=0 ; iter<iters ; iter++ )
			{
				for( unsigned int i=0 ; i<mcIndices.size() ; i++ )
				{
					ThreadPool::ParallelFor
						(
							0 , mcIndices[i].size() ,
							[&]( unsigned int thread , size_t j )
							{
								unsigned int idx = mcIndices[i][j];
								Polynomial::Polynomial1D< 4 > _Q = energy.quarticFit( x , idx , thread );
								for( unsigned int i=0 ; i<=4 ; i++ ) if( fabs( _Q.coefficient(i) )<EPS ) _Q.coefficient(i) = 0;
								Polynomial::Polynomial1D< 3 > _dQ = _Q.d(0);
								double roots[3];
								unsigned int cnt = Polynomial::Roots( _dQ , roots , EPS );
								if( !cnt ) MK_ERROR_OUT( "Expected roots!" );
								double s = roots[0];
								for( unsigned int i=1 ; i<cnt ; i++ ) if( s!=s || _Q(roots[i])<_Q(s) ) s = roots[i];
								if( s!=s ) s=0;
								x[idx] += s;
							}
						);
				}
			}

			if( verbosity>=2 )
			{
				if( !adaptive )
				{

					std::pair< double, double > _err = energy.energies(x);
					{
						Miscellany::StreamFloatPrecision sfp(std::cout, 5);
						std::cout << "\tError[" << std::setw(3) << (1 << depth) << "]: " << std::setw(9) << err << " -> " << (_err.first + _err.second) << " = " << _err.first << " + " << _err.second;
					}
				}
				else std::cout << "adaptive energy logging not implemented" << std::endl;
				std::cout << "\titers = " << iters;
				{
					Miscellany::StreamFloatPrecision sfp(std::cout, 2);
					std::cout << "\t" << std::setw(6) << timer.elapsed() << " (s)" << std::endl;
				}
			}
		}

		template< unsigned int Dim , typename SystemEnergyType >
		void GradientDescentSolve( unsigned int depth , const SystemEnergyType &energy , Eigen::VectorXd &x , unsigned int iters , int verbosity )
		{
			static const bool Sym = true;

			static_assert( std::is_base_of< ProductSystem::Energy< Dim , Sym , typename SystemEnergyType::Indexer > , SystemEnergyType >::value , "[ERROR] SystemEnergyType poorly formed" );
			static const double EPS = 1e-24;
			Miscellany::Timer timer;
			Hat::ScalarFunctions< Dim > scalarFunctions( 1<<depth );

			double err = 0.;
			if( verbosity>=2 ) err = energy( x );
			for( unsigned int iter=0 ; iter<iters ; iter++ )
			{
				Eigen::VectorXd d = energy.linearApproximation(x).l;
				Polynomial::Polynomial1D< 4 > Q = energy.quarticFit( x , d , false );
				Polynomial::Polynomial1D< 3 > dQ = Q.d(0);
				double roots[3];
				unsigned int cnt = Polynomial::Roots( dQ , roots , EPS );
				if( !cnt ) MK_ERROR_OUT( "Expected roots!" );
				double s = roots[0];
				for( unsigned int i=1 ; i<cnt ; i++ ) if( s!=s || Q(roots[i])<Q(s) ) s = roots[i];
				if( s!=s ) s=0;
				x += d * s;
			}

			if( verbosity>=2 )
			{
				std::pair< double , double > _err = energy.energies( x );
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 5 );
					std::cout << "\tError[" << std::setw(3) << (1<<depth) << "]: " << std::setw(9) << err << " -> " << ( _err.first+_err.second ) << " = " << _err.first << " + " << _err.second;
					std::cout << "\titers = " << iters;
				}
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 2 );
					std::cout << "\t" << std::setw(6) << timer.elapsed() << " (s)" << std::endl;
				}
			}
		}

		template< unsigned int Dim , typename SystemEnergyType >
		void NewtonSolve( unsigned int depth , const SystemEnergyType &energy , Eigen::VectorXd &x , unsigned int iters , int verbosity )
		{
			static const bool Sym = true;

			MK_WARN_ONCE( "Foregoing sanity test" );
			static const double EPS = 1e-24;
			Miscellany::Timer timer;
			Hat::ScalarFunctions< Dim > scalarFunctions( 1<<depth );
			Eigen::SimplicialLDLT< Eigen::SparseMatrix< double > > solver;
			double err = 0.;
			if( verbosity>=2 ) err = energy( x );
			for( unsigned int iter=0 ; iter<iters ; iter++ )
			{
				typename SystemEnergyType::QuadraticApproximation qa = energy.quadraticApproximation(x);
				if( !iter ) solver.analyzePattern( qa.q );
				solver.factorize( qa.q );
				if( solver.info()!=Eigen::Success ) MK_ERROR_OUT( "Failed to factorize matrix" );
				Eigen::VectorXd d = solver.solve( qa.l );
				Polynomial::Polynomial1D< 4 > Q = energy.quarticFit( x , d , false );
				Polynomial::Polynomial1D< 3 > dQ = Q.d(0);
				double roots[3];
				unsigned int cnt = Polynomial::Roots( dQ , roots , EPS );
				if( !cnt ) MK_ERROR_OUT( "Expected roots!" );
				double s = roots[0];
				for( unsigned int i=1 ; i<cnt ; i++ ) if( s!=s || Q(roots[i])<Q(s) ) s = roots[i];
				if( s!=s ) s=0;
				x += d * s;
			}

			if( verbosity>=2 )
			{
				std::pair< double , double > _err = energy.energies( x );
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 5 );
					std::cout << "\tError[" << std::setw(3) << (1<<depth) << "]: " << std::setw(9) << err << " -> " << ( _err.first+_err.second ) << " = " << _err.first << " + " << _err.second;
					std::cout << "\titers = " << iters;
				}
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 2 );
					std::cout << "\t" << std::setw(6) << timer.elapsed() << " (s)" << std::endl;
				}
			}
		}

#ifdef USE_NL_OPT
		template< unsigned int Dim , typename SystemEnergyType >
		double NLOptFunction( const std::vector< double > &x , std::vector< double > &grad , void *v )
		{
			const SystemEnergyType &energy = *( ( SystemEnergyType * )v );
			Eigen::VectorXd _x( x.size() );
			for( unsigned int i=0 ; i<x.size() ; i++ ) _x[i] = x[i];
			if( !grad.empty() ) 
			{
				Eigen::VectorXd d = energy.linearApproximation(_x).l;
				for( unsigned int i=0 ; i<d.size() ; i++ ) grad[i] = d[i];
			}
			return energy( _x );
		}

		template< unsigned int Dim , typename SystemEnergyType >
		void NLOptSolve( nlopt::algorithm algoType , unsigned int depth , const SystemEnergyType &energy , Eigen::VectorXd &x , unsigned int iters , unsigned int gradientMemory , int verbosity )
		{
			Miscellany::Timer timer;
			nlopt::opt opt( algoType , (unsigned int)x.size() );
			opt.set_min_objective( NLOptFunction< Dim , SystemEnergyType > , (void*)&energy );
			opt.set_maxeval( iters );
			//std::cout << std::string( opt.get_algorithm_name() ) << std::endl;
			if( gradientMemory!=-1 ) opt.set_vector_storage( gradientMemory );

			double err = 0.;
			if( verbosity>=2 ) err = energy( x );

			double e;
			std::vector< double > _x( x.size() );
			for( unsigned int i=0 ; i<x.size() ; i++ ) _x[i] = x[i];
			nlopt::result result = opt.optimize( _x , e );
			for( unsigned int i=0 ; i<x.size() ; i++ ) x[i] = _x[i];

			if( verbosity>=2 )
			{
				std::pair< double , double > _err = energy.energies( x );
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 5 );
					std::cout << "\tError[" << std::setw(3) << (1<<depth) << "]: " << std::setw(9) << err << " -> " << ( _err.first+_err.second ) << " = " << _err.first << " + " << _err.second;
					std::cout << "\titers = " << iters;
				}
				{
					Miscellany::StreamFloatPrecision sfp( std::cout , 2 );
					std::cout << "\t" << std::setw(6) << timer.elapsed() << " (s)" << std::endl;
				}
			}
		}
#endif // USE_NL_OPT

	}
}
#endif // POISSON_SOLVERS_INCLUDED