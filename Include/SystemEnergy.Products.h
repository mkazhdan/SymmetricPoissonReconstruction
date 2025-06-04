#ifndef SYSTEM_ENERGY_PRODUCTS_INCLUDED
#define SYSTEM_ENERGY_PRODUCTS_INCLUDED

#include "PreProcessor.h"
#include "Misha/Polynomial.h"

namespace MishaK
{
	namespace ProductSystem
	{
		// A structure describing the system energy
		// Used for solving systems of the form:
		//		E(x,y) = || x . y - T ||_F^2 + x^t * ( R + S ) * x + y^t * ( R + S ) * y
		// Where:
		//		"x.y" is either the alternating product of the gradients or the symmetric product of the gradients
		//		T is the target (skew-)symmetric tensor field
		//		R is a quadratic regularization energy, described by a sparse matrix
		//		S is a quadratic regularization energy, described by a stencil
		template< unsigned int Dim , bool Sym , typename _Indexer /* = Hat::BaseIndexer< Dim > */ >
		struct Energy
		{
			using Indexer = _Indexer;
			struct LinearApproximation
			{
				Eigen::VectorXd l;
				double c;

				LinearApproximation( void ) : c(0){}
				LinearApproximation( const Eigen::VectorXd  &l , double c ) : l(l) , c(c){}
				LinearApproximation( const Eigen::VectorXd &&l , double c ) : l( std::move(l) ) , c(c){}

				double operator()( const Eigen::VectorXd &x ) const { return l.dot(x) + c; }
			};

			struct QuadraticApproximation : public LinearApproximation
			{
				Eigen::SparseMatrix< double > q;

				QuadraticApproximation( void ) : LinearApproximation(){}
				QuadraticApproximation( const Eigen::SparseMatrix< double > &q , const Eigen::VectorXd &l , double c ) : LinearApproximation(l,c) , q(q){}
				QuadraticApproximation( Eigen::SparseMatrix< double > &&q , const Eigen::VectorXd &&l , double c ) : LinearApproximation(l,c) , q( std::move(q) ){}

				double operator()( const Eigen::VectorXd &x ) const { return ( q * x ).dot( x ) + LinearApproximation::operator()(x); }
			};

			Indexer indexer;
			Hat::ScalarFunctions< Dim > scalars;			// The scalar system
			Hat::ProductFunctions< Dim , Sym > products;	// The product system

			Energy( const Indexer &indexer , unsigned int r ) : indexer(indexer) , products(r) , scalars(r){}
			Energy( Energy &&se ) : Energy< Dim , Sym , Indexer >( indexer , 1 ){ std::swap( indexer , se.indexer ) , std::swap( scalars , se.scalars ) , std::swap( products , se.products ); }
			Energy &operator = ( Energy &&se ){ indexer = std::move( se.indexer ) , scalars = std::move( se.scalars ) , products = std::move( se.products ) ; return *this; }

			//////////////////////////////////
			// Evaluation and approximation //
			//////////////////////////////////

			// Computes the energy components given by the product-fitting and regularization energy terms
			double operator()( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			double operator()( const Eigen::VectorXd &x ) const;

			// Separately computes the energy components given by the alternating-field fitting (first) and regularization (second) energy terms
			virtual std::pair< double , double > energies( const Eigen::VectorXd &x ) const { return energies(x,x); }


			// Computes the differential of the energy with respect to the two function's coefficients
			virtual LinearApproximation linearApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Computes the differential of the energy with respect to the two function's coefficients
			virtual LinearApproximation linearApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Computes the differential of the energy with respect to the function's coefficients
			virtual LinearApproximation linearApproximation( const Eigen::VectorXd &x ) const;

			// Given the current estimate of the solution (x,y), gives the quadratic and linear approximations to the energy in x
			virtual QuadraticApproximation quadraticApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution (x,y), gives the quadratic and linear approximations to the energy in y
			virtual QuadraticApproximation quadraticApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution x, gives the quadratic and linear approximations to the energy in x
			virtual QuadraticApproximation quadraticApproximation( const Eigen::VectorXd &x ) const;


			// Given the current estimate of the solution (x,y) and the two directions (_x,_y), computes the value of s minimizing E(x+s*_x,y+s*_y)
			double stepSize( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &_x , const Eigen::VectorXd &_y ) const;

			// Given the current estimate of the solution (x,y) and the two directions (_x,_y), computes the values of (s,t) minimizing P(s,t) = E(x+s*_x,y+t*_y) using Newton iterations
			Point< double , 2 > newtonUpdate( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &dx , const Eigen::VectorXd &dy , unsigned int steps ) const;


			// Given the current estimate of the solution (x,y) computes the quadratic polynomial P(s,t) = E(x+s*e[idx],y+t*e[idx])
			// [NOTE] For asymmetric products, this should reduce to a quadratic polynomial
			virtual Polynomial::Polynomial2D< 4 > biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , size_t idx ) const;

			// Given the current estimate of the solution x computes the quadratic polynomial P(s) = E(x+s*e[idx],x+s*e[idx])
			virtual Polynomial::Polynomial1D< 4 > quarticFit( const Eigen::VectorXd &x , size_t idx , unsigned int thread ) const;

			// Given the current estimate of the solution (x,y) and the two directions (_x,_y), computes the bi-quadratic polynomial P(s,t) = E(x+s*_x,y+t*_y)
			virtual Polynomial::Polynomial2D< 4 > biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &_x , const Eigen::VectorXd &_y , bool setConstantTerm ) const;

			//////////////////////////
			// Pure virtual methods //
			//////////////////////////

			// Separately computes the energy components given by the alternating-field fitting (first) and regularization (second) energy terms
			virtual std::pair< double , double > energies( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const = 0;

			// Given the current estimate of the solution x computes the quadratic polynomial P(s) = E(s*x,s*y)
			virtual Polynomial::Polynomial1D< 4 > scalingQuarticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const = 0;

			// Returns the square norm of the target product
			virtual double targetSquareNorm( void ) const = 0;

			// Returns the square norm of the product
			virtual double squareNorm( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const = 0;

			// Returns the square norm of the product
			virtual double squareNorm( const Eigen::VectorXd &x ) const = 0;

			// Returns the prolongation of the scalar function
			virtual Eigen::VectorXd scalarProlongation( const Eigen::VectorXd &coarse ) const = 0;

		protected:
			Energy( void ) : Energy(1){}
		};

		template< unsigned int Dim , bool Sym >
		struct BasicEnergy// : public Energy< Dim , Sym >
		{
			//tex:
			// \begin{align*}
			// E(x,y) &= \| x\times y - \tau\|^2 + x^\top\cdot \mathbf{R}\cdot x + y^\top\cdot \mathbf{R}\cdot y\\
								//        &= (x \times y )^\top\cdot \mathbf{M}\cdot (x\times y) - 2(x\times y )^\top\cdot \mathbf{M}\cdot \tau + \tau^\top\cdot \mathbf{M}\cdot \tau + x^\top\cdot \mathbf{R}\cdot x + y^\top\cdot \mathbf{R}\cdot y\\
		//        &= (x \times y )^\top\cdot \mathbf{M}\cdot (x\times y) - 2 y^\top\cdot A_{M\cdot\tau}\cdot x + \tau^\top\cdot \mathbf{M}\cdot \tau + x^\top\cdot \mathbf{R}\cdot x + y^\top\cdot \mathbf{R}\cdot y\\
		// \end{align*}
// where $\mathbf{M}\in\mathbb{R}^{N\times N}$ is the mass matrix.

//tex:
// In what follows, given a vector $x\in\mathbb{R}^n$, we let $\mathbf{X}\in\mathbb{R}^{N\times n}$ be the matrix such that $\mathbf{X}\cdot y = x\times y$.
// Furthermore, we recall that for any $Z\in\mathbb{R}^N$, we have:
// $$\mathbf{X}^\top\cdot Z = \mathbf{A}_{Z}\cdot x.$$

//tex:
// Writing: 
// $$E(x,y) = y^\top\cdot \mathbf{X}^\top\cdot\mathbf{M}\cdot \mathbf{X}\cdot y - 2 y^\top\cdot\mathbf{X}^\top\cdot\mathbf{M}\cdot\tau + \tau^\top\cdot\mathbf{M}\cdot \tau + x^\top\cdot\mathbf{R}\cdot x + y^\top\cdot\mathbf{R}\cdot y.$$
// Differentiating with respect to $y$ gives:
// \begin{align*}
// \frac{dE}{dy} &= 2 \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X}\cdot y - 2 X^\top\cdot\mathbf{M}\cdot \tau + 2\mathbf{R}\cdot y\\
		//               &= 2 \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X}\cdot y - 2 \mathbf{A}_{\mathbf{M}\cdot\tau}\cdot x + 2\mathbf{R}\cdot y \\
 		//               &= \pm 2 \mathbf{X}^\top\cdot\mathbf{M}\cdot \mathbf{Y}\cdot x - 2 \mathbf{A}_{\mathbf{M}\cdot\tau}\cdot x + 2\mathbf{R}\cdot y \\
		// \end{align*}
// where the signs are selected as "symmetric/asymmetric".
// Taking the second derivative with respect to $y$ gives:
// $$\frac{d^2E}{dy^2} = 2\mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X} + 2\mathbf{R}.$$
// Intead, taking the derivative with respect to $x$ gives:
// $$\frac{d^2E}{dxdy} = \pm 2\mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{Y} + 2\mathbf{A}_{\mathbf{M}\cdot x\times y} - 2 \mathbf{A}_{\mathbf{M}\cdot\tau}.$$

//tex:
// Instead, writing: 
// $$E(x,y) = x^\top\cdot\mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y}\cdot x \mp 2 x^\top\cdot\mathbf{Y}^\top\cdot\mathbf{M}\cdot \tau + \tau^\top\cdot\mathbf{M}\cdot \tau + x^\top\cdot\mathbf{R}\cdot x + y^\top\cdot\mathbf{R}\cdot y.$$
// Differentiating with respect to $x$ gives:
// \begin{align*}
// \frac{dE}{dx} &= 2 \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y}\cdot x \mp 2\mathbf{Y}^\top\cdot\mathbf{M}\cdot\tau + 2\mathbf{R}\cdot x\\
		//               &= 2 \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y}\cdot x \mp 2 \mathbf{A}_{\mathbf{M}\cdot\tau}\cdot y + 2\mathbf{R}\cdot x \\
 		//               &= \pm 2 \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{X}\cdot y \mp 2 \mathbf{A}_{\mathbf{M}\cdot\tau}\cdot y + 2\mathbf{R}\cdot x \\
		// \end{align*}
// Taking the second derivative with respect to $x$ gives:
// $$\frac{d^2E}{dx^2} = 2 \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y} + 2\mathbf{R}.$$
// Intead, taking the derivative with respect to $y$ gives:
// $$\frac{d^2E}{dydx} = \pm 2 \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{X} \pm 2 \mathbf{A}_{\mathbf{M}\cdot x\times y} \mp 2 \mathbf{A}_{\mathbf{M}\cdot\tau}.$$

//tex:
// Putting all this together we get:
// \begin{align*}
// \frac{dE}{dx}       &= 2\left( \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y}\cdot x \mp \mathbf{Y}^\top\cdot\mathbf{M}\cdot \tau + \mathbf{R}\cdot x\right)\\
		// \frac{dE}{dy}       &= 2\left( \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X}\cdot y  -  \mathbf{X}^\top\cdot\mathbf{M}\cdot \tau + \mathbf{R}\cdot y\right)\\
		// \frac{d^2E}{dx^2}   &= 2\left( \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y} + \mathbf{R}\right)\\
		// \frac{d^2E}{dy^2}   &= 2\left( \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X} + \mathbf{R}\right)\\
		// \frac{d^2E}{dy\,dx} &= 2\left( \pm \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{X} \pm \mathbf{A}_{\mathbf{M}\cdot x\times y} \mp \mathbf{A}_{\mathbf{M}\cdot\tau}\right)\\
		// \frac{d^2E}{dx\,dy} &= 2\left( \pm \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{Y}  +  \mathbf{A}_{\mathbf{M}\cdot x\times y}  -  \mathbf{A}_{\mathbf{M}\cdot\tau}\right)
// \end{align*}

//tex:
// Given $x,y,\hat{x},\hat{y}\in\mathbb{R}^n$, we have the bi-quadratic polynomial:
// \begin{align*}
// E_{x,y,\hat{x},\hat{y}}(s,t)
// &= \| (x+s\hat{x})\times(y+t\hat{y}) - \tau\|^2 + (x+s\hat{x})^\top\cdot \mathbf{R}\cdot(x+s\hat{x}) + (y+t\hat{y})^\top\cdot \mathbf{R}\cdot (y+t\hat{y})\\
		// &= (x \times y )^\top\cdot \mathbf{M}\cdot (x\times y) - 2(x\times y )^\top\cdot \mathbf{M}\cdot \tau + \tau^\top\cdot \mathbf{M}\cdot \tau + x^\top\cdot \mathbf{R}\cdot x + y^\top\cdot \mathbf{R}\cdot y\\
		// &+ s\left(2(\hat{x}\times y)^\top\cdot\mathbf{M}\cdot(x\times y) + 2\hat{x}^\top\cdot\mathbf{R}\cdot x - 2(\hat{x}\times y)^\top\cdot\mathbf{M}\cdot\tau\right)\\
		// &+ t\left(2(x\times \hat{y})^\top\cdot\mathbf{M}\cdot(x\times y) + 2\hat{y}^\top\cdot\mathbf{R}\cdot y - 2(x\times\hat{y})^\top\cdot\mathbf{M}\cdot\tau\right)\\
		// &+ st\left(2(\hat{x}\times\hat{y})^\top\cdot\mathbf{M}\cdot(x\times y)+2(\hat{x}\times y)^\top\cdot\mathbf{M}\cdot(x\times\hat{y}) - 2(\hat{x}\times\hat{y})^\top\cdot\mathbf{M}\cdot\tau\right)\\
		// &+ s^2\left((\hat{x}\times y)^\top\cdot\mathbf{M}\cdot(\hat{x}\times y)+\hat{x}^\top\cdot\mathbf{R}\cdot\hat{x}\right)\\
		// &+ t^2\left((x\times\hat{y})^\top\cdot\mathbf{M}\cdot(x\times\hat{y})+\hat{y}^\top\cdot\mathbf{R}\cdot\hat{y}\right)\\
		// &+ s^2t\left(2(\hat{x}\times\hat{y})^\top\cdot\mathbf{M}\cdot(\hat{x}\times y)\right)\\
		// &+ st^2\left(2(\hat{x}\times\hat{y})^\top\cdot\mathbf{M}\cdot(x\times \hat{y})\right)\\
		// &+ s^2t^2\left((\hat{x}\times\hat{y})^\top\cdot\mathbf{M}\cdot(\hat{x}\times \hat{y})\right)
// \end{align*}


		protected:
			Hat::ScalarFunctions< Dim > _scalars;
			Hat::ProductFunctions< Dim , Sym > _products;
			Eigen::SparseMatrix< double > _M , _R;
			std::vector< Hat::SquareMatrix< double , Dim , Sym > > _t;
			Eigen::VectorXd _Mt;
			double _tSquareNorm;

		public:
			BasicEnergy( unsigned int res , ConstPointer( Hat::SquareMatrix< double , Dim , Sym > ) t , Eigen::SparseMatrix< double > R );
			std::pair< double , double > energies( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			double value( const Eigen::VectorXd &x , const Eigen::VectorXd &y , unsigned int samplesPerDim ) const;

			double operator()( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			double operator()( const Eigen::VectorXd &x ) const { return operator()(x,x); }

			Eigen::VectorXd dX( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			Eigen::VectorXd dY( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			Eigen::VectorXd d( const Eigen::VectorXd &x ) const { return dX(x,x) + dY(x,x); }


			Eigen::SparseMatrix< double > dXdX( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			Eigen::SparseMatrix< double > dYdY( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			Eigen::SparseMatrix< double > dXdY( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			Eigen::SparseMatrix< double > dYdX( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			Eigen::SparseMatrix< double > d2( const Eigen::VectorXd &x ) const { return dXdX(x,x) + dYdY(x,x) + dXdY(x,x) + dYdX(x,x); }

			Polynomial::Polynomial2D< 4 > biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &_x , const Eigen::VectorXd & _y ) const;
		};


		template< unsigned int Dim , bool Sym , typename Indexer /* = Hat::BaseIndexer< Dim > */  >
		struct CascadicSystemEnergy : public Energy< Dim , Sym , Indexer >
		{
			using Energy< Dim , Sym , Indexer >::indexer;
			using Energy< Dim , Sym , Indexer >::scalars;
			using Energy< Dim , Sym , Indexer >::products;

			// Construct the system energy

			template< typename MatrixField /* = std::function< Hat::SquareMatrix< double , Dim , Sym > ( Point< double , Dim > , usnigned int ) > */ >
			CascadicSystemEnergy( const Indexer & indexer , unsigned int r , MatrixField && mField , bool linearTensor , double sWeight , Eigen::SparseMatrix< double > R );

			// Restrict from the finer resolution
			CascadicSystemEnergy restrict( const Indexer &coarseProlongationIndexer ) const;


			// Update the system given the finer solution
			void update( const CascadicSystemEnergy &finer , const Eigen::VectorXd &x , const Eigen::VectorXd &y );

			// Separately computes the energy components given by the alternating-field fitting (first) and regularization (second) energy terms
			std::pair< double , double > energies( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;
			std::pair< double , double > energies( const Eigen::VectorXd &x ) const;

			// Given the current estimate of the solution x computes the quadratic polynomial P(s) = E(s*x,s*y)
			Polynomial::Polynomial1D< 4 > scalingQuarticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution x computes the quadratic polynomial P(s) = E(s*x,s*y)
			Polynomial::Polynomial1D< 4 > scalingQuarticFit( const Eigen::VectorXd &x ) const;

			// Given the current estimate of the solution (x,y) computes the quadratic polynomial P(s,t) = E(x+s*e[idx],y+t*e[idx])
			Polynomial::Polynomial2D< 4 > biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , size_t idx ) const;

			// Given the current estimate of the solution x computes the quadratic polynomial P(s) = E(x+s*e[idx],x+t*e[idx])
			Polynomial::Polynomial1D< 4 > quarticFit( const Eigen::VectorXd &x , size_t idx , unsigned int thread ) const;


			// Given the current estimate of the solution (x,y) and the two directions (_x,_y), computes the bi-quadratic polynomial P(s,t) = E(x+s*_x,y+t*_y)
			Polynomial::Polynomial2D< 4 > biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &_x , const Eigen::VectorXd &_y , bool setConstantTerm ) const;

			// Given the current estimate of the solution x and the direction _x, computes the quartic polynomial P(s) = E(x+s*_x)
			Polynomial::Polynomial1D< 4 > quarticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &_x , bool setConstantTerm ) const;


			// Returns the square norm of the target alternating product
			double targetSquareNorm( void ) const { return _c; }

			// Returns the square norm of the product
			double squareNorm( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const { return products( indexer , _pMStencil , x , y , x , y ); }

			// Returns the square norm of the product
			double squareNorm( const Eigen::VectorXd &x ) const;// { return products( _pMStencil , x , x , x , x ); }

			// Returns the prolongation of the scalar function
			Eigen::VectorXd scalarProlongation( const Eigen::VectorXd &coarse ) const { return _sP * coarse; }

			// Returns the scalar prolongation matrix
			const Eigen::SparseMatrix< double > &scalarProlongation( void ) const { return _sP; }

			// Returns a reference to the stiffness weight
			double &sWeight( void ){ return _sWeight; }
			const double &sWeight( void ) const { return _sWeight; }

			// Give access to the regularizer
			Eigen::SparseMatrix< double > &regularizer( void ){ return _R; }
			const Eigen::SparseMatrix< double > &regularizer( void ) const { return _R; }

			// Give access to the stiffness
			double & constant(void) { return _c; }
			const double & constant(void) const { return _c; }

			// Give access to the c
			Eigen::SparseMatrix< double >& stiffness(void) { return _B; }
			const Eigen::SparseMatrix< double >& stiffness(void) const { return _B; }

			// Given the current estimate of the solution (x,y), gives the quadratic and linear approximations to the energy in x
			typename Energy< Dim , Sym , Indexer >::QuadraticApproximation quadraticApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution (x,y), gives the quadratic and linear approximations to the energy in y
			typename Energy< Dim , Sym , Indexer >::QuadraticApproximation quadraticApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution (x,x), gives the quadratic and linear approximations to the energy in x
			typename Energy< Dim , Sym , Indexer >::QuadraticApproximation quadraticApproximation( const Eigen::VectorXd &x ) const;

			// Given the current estimate of the solution (x,y), gives the quadratic and linear approximations to the energy in x
			typename Energy< Dim , Sym , Indexer >::LinearApproximation linearApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution (x,y), gives the quadratic and linear approximations to the energy in y
			typename Energy< Dim , Sym , Indexer >::LinearApproximation linearApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const;

			// Given the current estimate of the solution (x,x), gives the quadratic and linear approximations to the energy in x
			typename Energy< Dim , Sym , Indexer >::LinearApproximation linearApproximation( const Eigen::VectorXd &x ) const;

		protected:
			CascadicSystemEnergy( const Indexer & indexer , unsigned int r );

			typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double > _pMStencil;	// The full product mass matrix stencil
			typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double > __pMStencil;	// The full product mass matrix stencil (with entries scaled based on whether or not they are diagonal)
			typename Hat::ScalarFunctions< Dim >::template FullIntegrationStencil< double , 0 > _sStencil;		// The full scalar matrix stencil
			// these will become the adaptive values version
			Eigen::SparseMatrix< double > _R;																	// The symmetric matrix defining the regularization energy
			Eigen::SparseMatrix< double > _sP;																	// The scalar prolongation matrix (from the coarse resolution into this one)
			Eigen::SparseMatrix< double > _B;																	// The integral of the target alternating-form field against the alternating-form basis
			double _c;																							// The constant terms coming from fitting
			double _sWeight;																					// The stencil weight
		};
#include "SystemEnergy.Products.inl"
	}
}
#endif // SYSTEM_ENERGY_PRODUCTS_INCLUDED