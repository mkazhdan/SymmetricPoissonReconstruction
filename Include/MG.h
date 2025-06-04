#ifndef MG_INCLUDED
#define MG_INCLUDED

#include <Eigen/Sparse>
#include "Misha/MultiThreading.h"


namespace MishaK
{
	struct GSRelaxer
	{
		GSRelaxer( const Eigen::SparseMatrix< double > &M ) : _M(M)
		{
			_D.resize( _M.rows() );
			for( unsigned int i=0 ; i<_M.outerSize() ; i++ ) for( Eigen::InnerIterator it(_M,i) ; it ; ++it )
				if( it.row()==it.col() ) _D[ it.row() ] = it.value();
		}

		void operator()( const Eigen::VectorXd &b , Eigen::VectorXd &x ) const
		{
			// \sum_j M_ij * x_j = b_i   ->   x_i <- ( \sum_j M_ij * x_j - b_i - D_i*x_i ) / D_i
			for( unsigned int i=0 ; i<_M.outerSize() ; i++ )
			{
				double value = 0;
				for( Eigen::InnerIterator it(_M,i) ; it ; ++it ) value += it.value() * x[ it.row() ];
				x[i] += ( b[i]  - value ) / _D[i];
			}
		}

		void operator()( const Eigen::VectorXd &b , Eigen::VectorXd &x , const std::vector< std::vector< unsigned int > > &mcIndices ) const
		{
			// \sum_j M_ij * x_j = b_i   ->   x_i <- ( \sum_j M_ij * x_j - b_i - D_i*x_i ) / D_i
			for( unsigned int i=0 ; i<mcIndices.size() ; i++ )
				ThreadPool::ParallelFor
				(
					0 , mcIndices[i].size() ,
					[&]( size_t j )
					{
						unsigned int idx = mcIndices[i][j];
						double value = 0;
						for( Eigen::InnerIterator it(_M,idx) ; it ; ++it ) value += it.value() * x[ it.row() ];
						x[idx] += ( b[idx]  - value ) / _D[idx];
					}
				);
		}
	protected:

		const Eigen::SparseMatrix< double > &_M;
		Eigen::VectorXd _D;
	};

	struct MGSolver
	{
		typedef Eigen::SimplicialLDLT< Eigen::SparseMatrix< double > > LDLt;

		template< typename ProlongationFunctor /* = std::function< Eigen::SparseMatrix< double > ( unsigned int ) > */ >
		MGSolver( const Eigen::SparseMatrix< double > &M , ProlongationFunctor Ps , size_t pNum ) : _M(M)
		{
			if( pNum )
			{
				_Ms.resize( pNum );
				_Ps.resize( pNum );
				_Rs.resize( pNum );
				_Xs.resize( pNum );
				_Bs.resize( pNum );
				for( unsigned int i=0 ; i<pNum ; i++ )
				{
					_Ps[i] = Ps( i );
					_Rs[i] = _Ps[i].transpose();
				}

				_Ms.back() = _Rs[pNum-1] * _M * _Ps[pNum-1];
				for( int i=(int)pNum-2 ; i>=0 ; i-- ) _Ms[i] = _Rs[i] * _Ms[i+1] * _Ps[i];
				for( int i=1 ; i<_Ms.size() ; i++ ) _gsRelaxers.emplace_back( _Ms[i] );
				_gsRelaxers.emplace_back( _M );
				for( unsigned int i=0 ; i<_Ms.size() ; i++ ) _Xs[i].resize( _Ms[i].cols() );

				_solver = new Eigen::SimplicialLDLT< Eigen::SparseMatrix< double > >( _Ms[0] );
			}
			else _solver = new Eigen::SimplicialLDLT< Eigen::SparseMatrix< double > >( _M );
		}

		~MGSolver( void ){ delete _solver; }

		void vCycle( const Eigen::VectorXd &b , Eigen::VectorXd &x , unsigned int iters )
		{
			// Fine-to-coarse
			for( unsigned int l=(unsigned int)_Ps.size() ; l>0 ; l-- )
			{
				if( l==_Ps.size() )
				{
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( b , x );
					_Bs[l-1] = _Rs[l-1] * ( b - _M * x );
					for( unsigned int i=0 ; i<_Xs[l-1].size() ; i++ ) _Xs[l-1][i] = 0;
				}
				else
				{
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( _Bs[l] , _Xs[l] );
					_Bs[l-1] = _Rs[l-1] * ( _Bs[l] - _Ms[l] * _Xs[l] );
					for( unsigned int i=0 ; i<_Xs[l-1].size() ; i++ ) _Xs[l-1][i] = 0;
				}
			}
			// Low-res solve
			if( _Ps.size() ) _Xs[0] = _solver->solve( _Bs[0] );
			else             x = _solver->solve( b );

			// Coarse-to-fine
			for( unsigned int l=1 ; l<=_Ps.size() ; l++ )
			{
				if( l==_Ps.size() )
				{
					x += _Ps[l-1] * _Xs[l-1];
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( b , x );
				}
				else
				{
					_Xs[l] += _Ps[l-1] * _Xs[l-1];
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( _Bs[l] , _Xs[l] );
				}
			}
		}

		template< typename MCIndicesFunctor /* = std::function< const std::vector< std::vector< unsigned int > > & ( unsigned int ) > */ >
		void vCycle( const Eigen::VectorXd &b , Eigen::VectorXd &x , unsigned int iters , MCIndicesFunctor MCI )
		{
			// Fine-to-coarse
			for( unsigned int l=(unsigned int)_Ps.size() ; l>0 ; l-- )
			{
				if( l==_Ps.size() )
				{
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( b , x , MCI(l) );
					_Bs[l-1] = _Rs[l-1] * ( b - _M * x );
					for( unsigned int i=0 ; i<_Xs[l-1].size() ; i++ ) _Xs[l-1][i] = 0;
				}
				else
				{
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( _Bs[l] , _Xs[l] , MCI(l) );
					_Bs[l-1] = _Rs[l-1] * ( _Bs[l] - _Ms[l] * _Xs[l] );
					for( unsigned int i=0 ; i<_Xs[l-1].size() ; i++ ) _Xs[l-1][i] = 0;
				}
			}
			// Low-res solve
			if( _Ps.size() ) _Xs[0] = _solver->solve( _Bs[0] );
			else             x = _solver->solve( b );

			// Coarse-to-fine
			for( unsigned int l=1 ; l<=_Ps.size() ; l++ )
			{
				if( l==_Ps.size() )
				{
					x += _Ps[l-1] * _Xs[l-1];
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( b , x , MCI(l) );
				}
				else
				{
					_Xs[l] += _Ps[l-1] * _Xs[l-1];
					for( unsigned int i=0 ; i<iters ; i++ ) _gsRelaxers[l-1]( _Bs[l] , _Xs[l] , MCI(l) );
				}
			}
		}

	protected:
		const Eigen::SparseMatrix< double > &_M;
		std::vector< Eigen::SparseMatrix< double > > _Ps , _Rs , _Ms;
		std::vector< GSRelaxer > _gsRelaxers;
		std::vector< Eigen::VectorXd > _Xs , _Bs;
		Eigen::SimplicialLDLT< Eigen::SparseMatrix< double > > *_solver;
	};
}
#endif // MG_INCLUDED