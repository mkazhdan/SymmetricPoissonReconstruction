////////////
// Energy //
////////////
template< unsigned int Dim , bool Sym , typename Indexer >
double Energy< Dim , Sym , Indexer >::operator()( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	std::pair< double , double > errs = energies( x , y );
	return errs.first + errs.second;
}

template< unsigned int Dim , bool Sym , typename Indexer >
double Energy< Dim , Sym , Indexer >::operator()( const Eigen::VectorXd &x ) const
{
	std::pair< double , double > errs = energies(x);
	return errs.first + errs.second;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::LinearApproximation Energy< Dim , Sym , Indexer >::linearApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	typename Energy< Dim , Sym , Indexer >::LinearApproximation la;
	la.c = operator()(x,y);
	la.l.resize( scalars.functionNum() );
	ThreadPool::ParallelFor
		(
			0 , scalars.functionNum() ,
			[&]( size_t i )
			{
				Polynomial::Polynomial2D< 4 > Q = biQuadraticFit( x , y , static_cast< unsigned int >(i) );
				la.l[i] = Q.coefficient(1,0);
			}
		);
	return la;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::LinearApproximation Energy< Dim , Sym , Indexer >::linearApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	typename Energy< Dim , Sym , Indexer >::LinearApproximation la;
	la.c = operator()(x,y);
	la.l.resize( scalars.functionNum() );
	ThreadPool::ParallelFor
		(
			0 , scalars.functionNum() ,
			[&]( size_t i )
			{
				Polynomial::Polynomial2D< 4 > Q = biQuadraticFit( x , y , i );
				la.l[i] = Q.coefficient(0,1);
			}
		);
	return la;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::LinearApproximation Energy< Dim , Sym , Indexer >::linearApproximation( const Eigen::VectorXd &x ) const
{
	typename Energy< Dim , Sym , Indexer >::LinearApproximation la;
	la.c = operator()(x,x);
	la.l.resize( scalars.functionNum() );
	ThreadPool::ParallelFor
		(
			0 , scalars.functionNum() ,
			[&]( size_t i )
			{
				Polynomial::Polynomial1D< 4 > Q = quarticFit( x , i , 0 );
				la.l[i] = Q.coefficient(1);
			}
		);
	return la;
}

template< unsigned int Dim , bool Sym , typename Indexer >
double Energy< Dim , Sym , Indexer >::stepSize( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &dx , const Eigen::VectorXd &dy ) const
{
	Matrix< double , 2 , 2 > M;
	M(0,0) = M(0,1) = 1.;

	Polynomial::Polynomial1D< 4 > Q = biQuadraticFit( x , y , dx , dy , false ).template pullBack< 2 >( M );
	Polynomial::Polynomial1D< 3 > dQ = Q.d(0);

	double roots[3];
	unsigned int rNum = Polynomial::Roots( dQ , roots );
	if( !rNum ) MK_ERROR_OUT( "Expected a root for an odd-degree polynomial" );
	double s = roots[0];
	for( unsigned int i=1 ; i<rNum ; i++ ) if( Q( roots[i] )<Q(s) ) s = roots[i];
	return s;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Point< double , 2 > Energy< Dim , Sym , Indexer >::newtonUpdate( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &dx , const Eigen::VectorXd &dy , unsigned int steps ) const
{
	Point< double , 2 > c;
	Polynomial::Polynomial2D< 4 > Q4 = biQuadraticFit( x , y , dx , dy , false );
	for( unsigned int step=0 ; step<steps ; step++ )
	{
		Point< double , 2 > d = - Q4.hessian( c ).inverse() * Q4.gradient( c );
		if( d.squareNorm()<1e-20 ) break;

		// Compute the 1D polynomial Q1(s) = cQ4( p * s );
		Polynomial::Polynomial1D< 4 > Q1;
		{
			Matrix< double , 2 , 2 > S;
			S(0,0) = d[0];
			S(0,1) = d[1];
			S(1,0) = c[0];
			S(1,1) = c[1];
			Q1 = Q4.template pullBack< 2 >( S );
		}
		Polynomial::Polynomial1D< 3 > dQ1 = Q1.d(0);

		double roots[3];
		unsigned int rNum = Polynomial::Roots( dQ1 , roots );
		if( !rNum ) MK_ERROR_OUT( "Expected a root for an odd-degree polynomial: " , Q1 );
		double s = roots[0];
		for( unsigned int i=1 ; i<rNum ; i++ ) if( Q1( roots[i] )<Q1(s) ) s = roots[i];
		c += d * s;
	}
	return c;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial2D< 4 > Energy< Dim , Sym , Indexer >::biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , size_t idx ) const
{
	Eigen::VectorXd d;
	d.setZero( x.size() );
	d[idx] = 1.;
	return biQuadraticFit( x , y , d , d , false );
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial1D< 4 > Energy< Dim , Sym , Indexer >::quarticFit( const Eigen::VectorXd &x , size_t idx , unsigned int thread ) const
{
	Eigen::VectorXd d;
	d.setZero( x.size() );
	d[idx] = 1.;
	Polynomial::Polynomial2D< 4 > _Q = biQuadraticFit( x , x , d , d , false );
	Polynomial::Polynomial1D< 4 > Q;
	Q.coefficient(4) = _Q.coefficient(2,2);
	Q.coefficient(3) = _Q.coefficient(1,3)+_Q.coefficient(3,1);
	Q.coefficient(2) = _Q.coefficient(0,2) + _Q.coefficient(1,1) + _Q.coefficient(2,0);
	Q.coefficient(1) = _Q.coefficient(0,1) + _Q.coefficient(1,0);
	return Q;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial2D< 4 > Energy< Dim , Sym , Indexer >::biQuadraticFit( const Eigen::VectorXd & , const Eigen::VectorXd & , const Eigen::VectorXd & , const Eigen::VectorXd & , bool ) const
{
	MK_ERROR_OUT( "Method not supported" );
	return Polynomial::Polynomial2D< 4 >();
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::QuadraticApproximation Energy< Dim , Sym , Indexer >::quadraticApproximation1( const Eigen::VectorXd & , const Eigen::VectorXd & ) const
{
	MK_ERROR_OUT( "Method not supported" );
	return typename Energy< Dim , Sym , Indexer >::QuadraticApproximation();
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::QuadraticApproximation Energy< Dim , Sym , Indexer >::quadraticApproximation2( const Eigen::VectorXd & , const Eigen::VectorXd & ) const
{
	MK_ERROR_OUT( "Method not supported" );
	return typename Energy< Dim , Sym , Indexer >::QuadraticApproximation();
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::QuadraticApproximation Energy< Dim , Sym , Indexer >::quadraticApproximation( const Eigen::VectorXd & ) const
{
	MK_ERROR_OUT( "Method not supported" );
	return typename Energy< Dim , Sym , Indexer >::QuadraticApproximation();
}

/////////////////
// BasicEnergy //
/////////////////
template< unsigned int Dim , bool Sym >
BasicEnergy< Dim , Sym >::BasicEnergy( unsigned int res , ConstPointer( Hat::SquareMatrix< double , Dim , Sym > ) t , Eigen::SparseMatrix< double > R )
	: _scalars(res) , _products(res) , _R(R)
{
	_M = _products.mass();
	_t.resize( _scalars.elementNum() );
	for( unsigned int i=0 ; i<_scalars.elementNum() ; i++ ) _t[i] = t[i];
	_Mt = _products.valueDual( [&]( size_t i ){ return t[i]; } );
	_tSquareNorm = 0;
	for( size_t i=0 ; i<_scalars.elementNum() ; i++ ) _tSquareNorm += t[i].squareNorm();
	for( unsigned int d=0 ; d<Dim ; d++ ) _tSquareNorm /= res;
}

template< unsigned int Dim , bool Sym >
double BasicEnergy< Dim , Sym >::operator()( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// \begin{align*}
	// E(x,y) &= \| x\times y - \tau\|^2 + x^\top\cdot \mathbf{R}\cdot x + y^\top\cdot \mathbf{R}\cdot y\\
			//        &= (x \times y )^\top\cdot \mathbf{M}\cdot (x\times y) - 2(x\times y )^\top\cdot \mathbf{M}\cdot \tau + \tau^\top\cdot \mathbf{M}\cdot \tau + x^\top\cdot \mathbf{R}\cdot x + y^\top\cdot \mathbf{R}\cdot y\\
		// \end{align*}

	Eigen::VectorXd xy = _products.product(x,y);
	return (_M*xy).dot(xy) - 2 * _Mt.dot(xy) + _tSquareNorm + (_R*x).dot(x) + (_R*y).dot(y);
}

template< unsigned int Dim , bool Sym >
std::pair< double , double > BasicEnergy< Dim , Sym >::energies( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	Eigen::VectorXd xy = _products.product(x,y);
	return std::pair< double , double >( (_M*xy).dot(xy) - 2 * _Mt.dot(xy) + _tSquareNorm , (_R*x).dot(x) + (_R*y).dot(y) );
}

template< unsigned int Dim , bool Sym >
double BasicEnergy< Dim , Sym >::value( const Eigen::VectorXd &x , const Eigen::VectorXd &y , unsigned int samplesPerDim ) const
{
	Hat::Range< Dim > sampleRange;
	for( unsigned int d=0 ; d<Dim ; d++ ) sampleRange.second[d] = samplesPerDim;
	double v = 0;

	auto f = [&]( Hat::Index< Dim > I )
		{
			Point< double , Dim > p;
			for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( I[d] + 0.5 ) / samplesPerDim;
			Point< double , Dim > g1 = _scalars.gradient( x , p );
			Point< double , Dim > g2 = _scalars.gradient( y , p );

			Hat::SquareMatrix< double , Dim , Sym > t;
			{
				Hat::Index< Dim > E;
				p *= _scalars.resolution();
				for( unsigned int d=0 ; d<Dim ; d++ ) E[d] = (int)p[d];
				t = _t[ _scalars.elementIndex( E ) ];
			}
			v += ( t - Hat::SquareMatrix< double , Dim , Sym >( g1 , g2 ) ).squareNorm();
		};
	sampleRange.process( f );

	for( unsigned int d=0 ; d<Dim ; d++ ) v /= samplesPerDim;
	return v + (_R*x).dot(x) + (_R*y).dot(y);
}

template< unsigned int Dim , bool Sym >
Eigen::VectorXd BasicEnergy< Dim , Sym >::dX( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// $$\frac{dE}{dx} = 2 Y^\top\cdot M\cdot Y\cdot x \mp 2 Y^\top\cdot M \cdot \tau + 2 R\cdot x$$

	Eigen::SparseMatrix< double > Y = _products.product(y);
	if constexpr( Sym ) return 2 * Y.transpose() * _M * Y * x - 2 * Y.transpose() * _Mt + 2 * _R * x;
	else                return 2 * Y.transpose() * _M * Y * x + 2 * Y.transpose() * _Mt + 2 * _R * x;
}

template< unsigned int Dim , bool Sym >
Eigen::VectorXd BasicEnergy< Dim , Sym >::dY( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// $$\frac{dE}{dy} = 2 X^\top\cdot M\cdot X\cdot y - 2 X^\top\cdot M \cdot \tau + 2 R\cdot y$$

	Eigen::SparseMatrix< double > X = _products.product(x);
	return 2 * X.transpose() * _M * X * y - 2 * X.transpose() * _Mt + 2 * _R * y;
}

template< unsigned int Dim , bool Sym >
Eigen::SparseMatrix< double > BasicEnergy< Dim , Sym >::dXdX( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// $$\frac{d^2E}{dx^2} = 2 Y^\top\cdot M\cdot Y + 2 R$$

	Eigen::SparseMatrix< double > Y = _products.product(y);
	return 2 * Y.transpose() * _M * Y + 2 * _R;
}

template< unsigned int Dim , bool Sym >
Eigen::SparseMatrix< double > BasicEnergy< Dim , Sym >::dYdY( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// $$\frac{d^2E}{dy^2} = 2 X^\top\cdot M\cdot X + 2 R$$

	Eigen::SparseMatrix< double > X = _products.product(x);
	return 2 * X.transpose() * _M * X + 2 * _R;
}

template< unsigned int Dim , bool Sym >
Eigen::SparseMatrix< double > BasicEnergy< Dim , Sym >::dYdX( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// $$\frac{d^2E}{dydx} = \pm 2 Y^\top\cdot M\cdot X \pm 2 A_{M\cdot x\times y} \mp 2 A_{M\cdot\tau}$$

	Eigen::SparseMatrix< double > X = _products.product(x) , Y = _products.product(y);
	Eigen::VectorXd _Mxy = _M * _products.product(x,y);
	if constexpr( Sym ) return   2 * Y.transpose() * _M * X + 2 * _products.toMatrix( _Mxy ) - 2 * _products.toMatrix( _Mt );
	else                return - 2 * Y.transpose() * _M * X - 2 * _products.toMatrix( _Mxy ) + 2 * _products.toMatrix( _Mt );
}

template< unsigned int Dim , bool Sym >
Eigen::SparseMatrix< double > BasicEnergy< Dim , Sym >::dXdY( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// $$\frac{d^2E}{dxdy} = \pm 2 X^\top\cdot M\cdot Y + 2 A_{M\cdot x\times y} - 2 A_{M\cdot\tau}$$

	Eigen::SparseMatrix< double > X = _products.product(x) , Y = _products.product(y);
	Eigen::VectorXd _Mxy = _M * _products.product(x,y);
	if constexpr( Sym ) return   2 * X.transpose() * _M * Y + 2 * _products.toMatrix( _Mxy ) - 2 * _products.toMatrix( _Mt );
	else                return - 2 * X.transpose() * _M * Y + 2 * _products.toMatrix( _Mxy ) - 2 * _products.toMatrix( _Mt );
}


template< unsigned int Dim , bool Sym >
Polynomial::Polynomial2D< 4 > BasicEnergy< Dim , Sym >::biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &_x , const Eigen::VectorXd & _y ) const
{
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

	Polynomial::Polynomial2D< 4 > Q;

	Eigen::SparseMatrix< double > M = _products.mass();
	Eigen::VectorXd xy = _products.product(x,y);
	Eigen::VectorXd _xy = _products.product(_x,y);
	Eigen::VectorXd x_y = _products.product(x,_y);
	Eigen::VectorXd _x_y = _products.product(_x,_y);

	Q.coefficient(0,0) = operator()( x , y );
	Q.coefficient(1,0) = 2*(M*_xy).dot(xy) + 2*(_R*_x).dot(x) - 2 * _xy.dot(_Mt);
	Q.coefficient(0,1) = 2*(M*x_y).dot(xy) + 2*(_R*_y).dot(y) - 2 * x_y.dot(_Mt);
	Q.coefficient(1,1) = 2*(M*_x_y).dot(xy) + 2*(M*_xy).dot(x_y) - 2 * _x_y.dot(_Mt);
	Q.coefficient(2,0) =   (M*_xy).dot(_xy) + (_R*_x).dot(_x);
	Q.coefficient(0,2) =   (M*x_y).dot(x_y) + (_R*_y).dot(_y);
	Q.coefficient(2,1) = 2*(M*_x_y).dot(_xy);
	Q.coefficient(1,2) = 2*(M*_x_y).dot(x_y);
	Q.coefficient(2,2) =   (M*_x_y).dot(_x_y);

	return Q;

}

//////////////////////////
// CascadicSystemEnergy //
//////////////////////////
template< unsigned int Dim , bool Sym , typename Indexer>
CascadicSystemEnergy< Dim , Sym , Indexer >::CascadicSystemEnergy( const Indexer & indexer , unsigned int r )
	: Energy< Dim , Sym , Indexer >( indexer , r ) , _pMStencil( Hat::ProductFunctions< Dim , Sym >::MassStencil(r) , r ) , __pMStencil( Hat::ProductFunctions< Dim , Sym >::MassStencil(r) , r ) , _sStencil( Hat::ScalarFunctions< Dim >::StiffnessStencil(r) , r ) ,  _c(0) , _sWeight(0)
{
	for( unsigned int i=0 ; i<__pMStencil.rows().size() ; i++ )
		for( unsigned int j=0 ; j<__pMStencil.rows()[i].size() ; j++ )
		{
			double scale = __pMStencil.rows()[i][j]._f2==Window::IsotropicIndex< Dim , 3 >::template I< 1 >() ? 1 : 2;
			for( unsigned int k=0 ; k<__pMStencil.rows()[i][j].entries.size() ; k++ )
			{
				if( __pMStencil.rows()[i][j].entries[k]._g1!=__pMStencil.rows()[i][j].entries[k]._g2 ) __pMStencil.rows()[i][j].entries[k].value *= scale*2;
				else                                                                                   __pMStencil.rows()[i][j].entries[k].value *= scale;
			}
		}
}

template< unsigned int Dim , bool Sym , typename Indexer >
template< typename MatrixField /* = std::function< Hat::SquareMatrix< double , Dim , Sym > ( Point< double , Dim > , usnigned int ) > */ >
CascadicSystemEnergy< Dim , Sym , Indexer >::CascadicSystemEnergy( const Indexer & indexer , unsigned int r , MatrixField && mField , bool linearTensor , double sWeight , Eigen::SparseMatrix< double > R )
	: CascadicSystemEnergy< Dim , Sym , Indexer >( indexer , r )
{
	_sWeight = sWeight;
	if( linearTensor )
	{
		auto Kernel = [&]( Hat::Index< Dim > F , unsigned int t )
			{
				Point< double , Dim > p;
				for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = static_cast< double >( F[d] )/r;
				return mField( p , t )();
			};
		_B = scalars.stiffness( indexer , Kernel , true );
	}
	else
	{
		auto Kernel = [&]( Hat::Index< Dim > E , unsigned int t )
			{
				Point< double , Dim > p;
				for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d] + 0.5 )/r;
				return mField( p , t )();
			};
		_B = scalars.stiffness( indexer , Kernel );
	}
	_R = R;
	ThreadPool::ParallelFor
		(
			0 , indexer.numElements() ,
			[&]( unsigned int thread , size_t  e )
			{
				Hat::Index< Dim > E = indexer.elementIndex(e);
				Point< double , Dim > p;
				for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d] + 0.5 ) / r;
				Atomic< double >::Add( _c , mField( p , thread ).squareNorm() );
			}
		);
	_c /= scalars.elementNum();
}

template< unsigned int Dim, bool Sym , typename Indexer >
CascadicSystemEnergy< Dim, Sym , Indexer > CascadicSystemEnergy< Dim, Sym , Indexer >::restrict( const Indexer &coarseProlongationIndexer ) const
{
	static_assert( std::is_base_of_v< Hat::BaseProlongationIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );
	if( scalars.resolution()&1 ) MK_ERROR_OUT( "Expected even resolution: " , scalars.resolution() );
	CascadicSystemEnergy coarser( coarseProlongationIndexer , scalars.resolution() / 2 );
	coarser._sP = coarser.scalars.prolongation( coarseProlongationIndexer , indexer.numFunctions() );
	coarser._R = coarser._sP.transpose() * _R * coarser._sP;
	coarser._B = coarser._sP.transpose() * _B * coarser._sP;
	coarser._c = _c;
	coarser._sWeight = _sWeight;
	return coarser;
}

template< unsigned int Dim , bool Sym , typename Indexer >
void CascadicSystemEnergy< Dim , Sym , Indexer >::update( const CascadicSystemEnergy &finer , const Eigen::VectorXd &x , const Eigen::VectorXd &y )
{
	if( finer.scalars.resolution()!=scalars.resolution()*2 ) MK_ERROR_OUT( "Resolutions don't match: " , finer.scalars.resolution() , " != 2 * " , scalars.resolution() );
	if( !( x.isZero() && y.isZero() ) ) MK_ERROR_OUT( "Non-trivial restriction not supported" );
}

template< unsigned int Dim , bool Sym , typename Indexer >
std::pair< double , double > CascadicSystemEnergy< Dim , Sym , Indexer >::energies( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{ 
	// E(x,y) = || dx ^ dy - b ||^2  + x^t * R * x + y^t * R * y 
	//        = || dx ^ dy ||^2 - 2 < dx ^ dy , b > + x^t * R * x + y^t * R * y
	std::pair< double , double > e;

	e.first = products( indexer , _pMStencil , x , y , x , y ) - 2. * y.dot( _B * x ) + _c;
	e.second = ( _R * x ).dot( x ) + ( _R * y ).dot( y );
	if( _sWeight ) e.second += ( scalars( indexer , _sStencil , x , x ) + scalars( indexer , _sStencil , y , y ) ) * _sWeight;
	return e;
}

template< unsigned int Dim , bool Sym , typename Indexer >
std::pair< double , double > CascadicSystemEnergy< Dim , Sym , Indexer >::energies( const Eigen::VectorXd &x ) const
{
	// E(x,y) = || dx ^ dy - b ||^2  + x^t * R * x + y^t * R * y 
	//        = || dx ^ dy ||^2 - 2 < dx ^ dy , b > + x^t * R * x + y^t * R * y
	std::pair< double , double > e;
	
	e.first = squareNorm( x ) - 2. * x.dot( _B * x ) + _c;
	e.second = 2 * ( _R * x ).dot( x );
	if( _sWeight ) e.second += 2 * scalars( indexer , _sStencil , x , x ) * _sWeight;
	return e;
}

// Given the current estimate of the solution x computes the quadratic polynomial P(s) = E(s*x,s*y)
template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial1D< 4 > CascadicSystemEnergy< Dim , Sym , Indexer >::scalingQuarticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	// E(x,y) = || dx ^ dy - b ||^2  + x^t * R * x + y^t * R * y 
	//        = || dx ^ dy ||^2 - 2 < dx ^ dy , b > + x^t * R * x + y^t * R * y
	std::pair< double , double > e;

	Polynomial::Polynomial1D< 4 > q;
	q.coefficient(4) = products( indexer , _pMStencil , x , y , x , y );
	q.coefficient(2) = -2 * y.dot( _B * x ) + ( _R * x ).dot( x ) + ( _R * y ).dot( y );
	if( _sWeight ) q.coefficient(2) += ( scalars( indexer , _sStencil , x , x ) + scalars( indexer , _sStencil , y , y ) ) * _sWeight;
	q.coefficient(0) = _c;
	return q;
}

template< unsigned int Dim , bool Sym , typename Indexer>
Polynomial::Polynomial1D< 4 > CascadicSystemEnergy< Dim , Sym , Indexer >::scalingQuarticFit( const Eigen::VectorXd &x ) const
{
	// E(s) = || s^2 * dx ^ dx - b ||^2  + 2 * s^2 * x^t * R * x
	//      = s^4 * || dx ^ dx ||^2 - 2 * s^2 * < dx ^ dx , b > + || b ||^2 + 2* s^2 * x^t * R * x
	std::pair< double , double > e;

	Polynomial::Polynomial1D< 4 > q;
	q.coefficient(4) = products( indexer , _pMStencil , x , x , x , x );
	q.coefficient(2) = -2 * x.dot( _B * x ) + 2 * ( _R * x ).dot( x );
	if( _sWeight ) q.coefficient(2) += 2 * scalars( indexer , _sStencil , x , x ) * _sWeight;
	q.coefficient(0) = _c;
	return q;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial2D< 4 > CascadicSystemEnergy< Dim , Sym , Indexer >::biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , const Eigen::VectorXd &_x , const Eigen::VectorXd &_y , bool setConstantTerm ) const
{
	Polynomial::Polynomial2D< 4 > Q;

	// Using "a.b" to denote either the symmetric product or the alternating product of a and b:
	//	 E(a,b) = || ( a _x + x ).( b _y + y ) - T ||^2
	//	        = || ab _x._y + a _x.y + b x._y + x.y - T ||^2
	// 1.		=   a^2 b^2 < _x._y , _x._y >
	// 2. 		+   a^2     < _x.y , _x.y >
	// 3.		+       b^2 < x._y , x._y >
	// 4.		+           < x.y , x.y >
	// 5.		+           < T , T >
	// 6.		+ 2 a^2 b   < _x._y , _x.y >
	// 7.		+ 2 a   b^2 < _x._y , x._y >
	// 8.		+ 2 a   b   < _x._y , x.y >
	// 9.		+ 2 a   b   < _x.y , x._y >
	// 10.		+ 2 a       < _x.y , x.y >
	// 11.		+ 2     b   < x._y , x.y >
	// 12.		- 2         < x.y , T >
	// 13.		- 2 a   b   < _x._y , T >
	// 14.		- 2 a       < _x.y , T >
	// 15.		- 2     b   < x._y , T >

	// 1.        =   a^2 b^2 < _x._y , _x._y >
	Q.coefficient(2,2) +=     products( indexer , _pMStencil , _x , _y , _x ,_y );
	// 6.		+ 2 a^2 b   < _x._y , _x.y >
	Q.coefficient(2,1) += 2 * products( indexer , _pMStencil , _x , _y , _x , y );
	// 7.		+ 2 a   b^2 < _x._y , x._y >
	Q.coefficient(1,2) += 2 * products( indexer , _pMStencil , _x , _y , x , _y );
	// 2. 		+   a^2     < _x.y , _x.y >
	Q.coefficient(2,0) +=     products( indexer , _pMStencil , _x , y , _x , y );
	// 3.		+       b^2 < x._y , x._y >
	Q.coefficient(0,2) +=     products( indexer , _pMStencil , x , _y , x , _y );
	// 8.		+ 2 a   b   < _x._y , x.y >
	// 9.		+ 2 a   b   < _x.y , x._y >
	Q.coefficient(1,1) += 2 * products( indexer , _pMStencil , _x , _y , x , y ) + 2 * products( indexer , _pMStencil , _x , y , x , _y );
	// 10.		+ 2 a       < _x.y , x.y >
	Q.coefficient(1,0) += 2 * products( indexer , _pMStencil , _x , y , x , y );
	// 11.		+ 2     b   < x._y , x.y >
	Q.coefficient(0,1) += 2 * products( indexer , _pMStencil , x , _y , x , y );
	// 4.		+           < x.y , x.y >
	if( setConstantTerm ) Q.coefficient(0,0) +=     products( indexer , _pMStencil , x , y , x , y );
	// 5.		+           < T , T >
	if( setConstantTerm ) Q.coefficient(0,0) += _c;

	// 13.		- 2 a   b   < _x._y , T >
	Q.coefficient(1,1) += - 2. * _y.dot( _B * _x );
	// 14.		- 2 a       < _x.y , T >
	Q.coefficient(1,0) += - 2. * y.dot( _B * _x );
	// 15.		- 2     b   < x._y , T >
	Q.coefficient(0,1) += - 2. * _y.dot( _B * x );
	// 12.		- 2         < x.y , T >
	if( setConstantTerm ) Q.coefficient(0,0) += - 2. * y.dot( _B * x );

	Q.coefficient(2,0) +=      ( _R * _x ).dot( _x );
	Q.coefficient(1,0) += 2. * ( _R * x ).dot( _x );
	if( setConstantTerm ) Q.coefficient(0,0) +=      ( _R * x ).dot( x );

	Q.coefficient(0,2) +=      ( _R * _y ).dot( _y );
	Q.coefficient(0,1) += 2. * ( _R * y ).dot( _y );
	if( setConstantTerm ) Q.coefficient(0,0) +=      ( _R * y ).dot( y );

	if( _sWeight )
	{
		Q.coefficient(2,0) +=      scalars( indexer , _sStencil , _x , _x ) * _sWeight;
		Q.coefficient(1,0) += 2. * scalars( indexer , _sStencil , x , _x ) * _sWeight;
		if( setConstantTerm ) Q.coefficient(0,0) += scalars( indexer , _sStencil , x , x ) * _sWeight;

		Q.coefficient(0,2) +=      scalars( indexer , _sStencil , _y , _y ) * _sWeight;
		Q.coefficient(0,1) += 2. * scalars( indexer , _sStencil , y , _y ) * _sWeight;
		if( setConstantTerm ) Q.coefficient(0,0) += scalars( indexer , _sStencil , y , y ) * _sWeight;
	}

	return Q;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial1D< 4 > CascadicSystemEnergy< Dim , Sym , Indexer >::quarticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &_x , bool setConstantTerm ) const
{
	//tex:
	// Given $x,\hat{x},\in\mathbb{R}^n$, we have the quartic polynomial:
	// \begin{align*}
	// E_{x,\hat{x}}(s)
	// &= \| (x+s\hat{x})\times(x+s\hat{x}) - \tau\|^2 + (x+s\hat{x})^\top\cdot \mathbf{R}\cdot(x+s\hat{x}) + (x+s\hat{x})^\top\cdot \mathbf{R}\cdot (x+s\hat{x})\\
	// &= s^0\left((x \times x )^\top\cdot \mathbf{M}\cdot (x\times x) - 2(x\times x )^\top\cdot \mathbf{M}\cdot \tau + \tau^\top\cdot \mathbf{M}\cdot \tau + 2x^\top\cdot \mathbf{R}\cdot x\right)\\
	// &+ s^1\left(4(\hat{x}\times x)^\top\cdot\mathbf{M}\cdot(x\times x) + 4\hat{x}^\top\cdot\mathbf{R}\cdot x - 4(\hat{x}\times x)^\top\cdot\mathbf{M}\cdot\tau\right)\\
	// &+ s^2\left(2(\hat{x}\times\hat{x})^\top\cdot\mathbf{M}\cdot(x\times x)+4(\hat{x}\times x)^\top\cdot\mathbf{M}\cdot(x\times\hat{x}) - 2(\hat{x}\times\hat{x})^\top\cdot\mathbf{M}\cdot\tau+2\hat{x}^\top\cdot\mathbf{R}\cdot\hat{x}\right)\\
	// &+ s^3\left(4(\hat{x}\times\hat{x})^\top\cdot\mathbf{M}\cdot(\hat{x}\times x)\right)\\
	// &+ s^4\left((\hat{x}\times\hat{x})^\top\cdot\mathbf{M}\cdot(\hat{x}\times \hat{x})\right)
	// \end{align*}

	Polynomial::Polynomial1D< 4 > q;
	if( setConstantTerm ) q.coefficient(0) = products( indexer , _pMStencil , x , x , x , x ) - 2 * ( _B*x ).dot(x) + _c + 2 * ( _R*x ).dot(x);
	q.coefficient(1) = 4 * products( indexer , _pMStencil , _x , x , x , x ) + 4 * (_R*x).dot(_x) - 4 * ( _B*_x).dot(x);
	q.coefficient(2) = 2 * products( indexer , _pMStencil , _x , _x , x , x ) + 4 * products( indexer , _pMStencil , _x , x , _x , x ) - 2 * ( _B*_x ).dot(_x) + 2 * (_R*_x).dot(_x);
	q.coefficient(3) = 4 * products( indexer , _pMStencil , _x , _x , _x , x );
	q.coefficient(4) =     products( indexer , _pMStencil , _x , _x , _x , _x );

	if( _sWeight )
	{
		if( setConstantTerm ) q.coefficient(0) += 2 * scalars( indexer , _sStencil , x , x ) * _sWeight;
		q.coefficient(1) += 4 * scalars( indexer , _sStencil , x , _x ) * _sWeight;
		q.coefficient(2) += 2 * scalars( indexer , _sStencil , _x , _x ) * _sWeight;
	}

	return q;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial2D< 4 > CascadicSystemEnergy< Dim , Sym , Indexer >::biQuadraticFit( const Eigen::VectorXd &x , const Eigen::VectorXd &y , size_t idx ) const
{
	Polynomial::Polynomial2D< 4 > Q;

	// E(a) = ( x + s*e_i )^t * R * ( x + s*e_i )
	//      = x^t * R * x + 2s x^t * R * e_i + s^2 e_i^t * R * e_i

	// Get the contribution from the regulrizer matrix
	for( Eigen::InnerIterator it(_R,idx) ; it ; ++it )
	{
		if( it.row()==idx )
		{
			Q.coefficient(2,0) += it.value();
			Q.coefficient(0,2) += it.value();
		}
		Q.coefficient(1,0) += 2. * it.value() * x[ it.row() ];
		Q.coefficient(0,1) += 2. * it.value() * y[ it.row() ];
	}

	// Get the contribution from the regularizer stencil
	if( _sWeight )
	{
		Hat::Index< Dim > f1 = scalars.functionIndex( idx );
		const typename Hat::ScalarFunctions< Dim >::template FullIntegrationStencil< double , 0 >::Row &row = _sStencil.row( f1 );
		for( unsigned int i=0 ; i<row.size() ; i++ )
		{
			double value = std::get<2>( row[i] ) * _sWeight;
			size_t _idx = scalars.functionIndex( f1 + std::get<0>(row[i]) );
			if( idx==_idx )
			{
				Q.coefficient(2,0) += value;
				Q.coefficient(0,2) += value;
			}
			Q.coefficient(1,0) += 2. * value * x[_idx];
			Q.coefficient(0,1) += 2. * value * y[_idx];
		}
	}

	///////////////////////////////////////
	// Using "a.b" to denote either the symmetric product or the alternating product of a and b:
	//	 E(a,b) = || ( a e_i + x ).( b e_i + y ) - T ||^2
	//	        = || ab e_i.e_i + a e_i.y + b x.e_i + x.y - T ||^2
	// Expanding out we get:
	// 1.       =     a^2*b^2 < e_i.e_i , e_i.e_i >
	// 2.		+   2 a^2*b   < e_i.e_i , e_i.y >		< e_i.e_i , e_i.y >	= \sum_j    y[j]                < e_i.e_i , e_i.e_j >
	// 3.		+/- 2 a  *b^2 < e_i.e_i , e_i.x >		< e_i.e_i , e_i.x >	= \sum_j    x[j]                < e_i.e_i , e_i.e_j >
	// 4.		+   2 a  *b   < e_i.e_i , x.y >			< e_i.e_i , x.y >	= \sum_jk   x[j]*y[k]           < e_i.e_i , e_j.e_k >
	// 5.		-   2 a  *b   < e_i.e_i , T >			< e_i.e_i , T >		=                               < e_i.e_i , T > = e_i.dot( _B * e_i )
	// 6.		+     a^2     < e_i.y , e_i.y >			< e_i.y , e_i.y >	= \sum_jk   y[j]*y[k]           < e_i.e_j , e_i.e_k >
	// 7.		+     b^2     < e_i.x , e_i.x >			< e_i.x , e_i.x >	= \sum_jk   x[j]*x[k]           < e_i.e_j , e_i.e_k >
	// 8.		+/- 2 a  *b   < e_i.y , e_i.x >			< e_i.y , e_i.x >	= \sum_jk   x[j]*y[k]           < e_i.e_j , e_i.e_k >
	// 9.		+   2 a       < e_i.y , x.y >			< e_i.y , x.y >		= \sum_jkl  y[j]*x[k]*y[l]      < e_i.e_j , e_k.e_l >
	// 10.		+/- 2 b       < e_i.x , x.y >			< e_i.x , x.y >		= \sum_jkl  x[j]*x[k]*y[l]      < e_i.e_j , e_k.e_l >
	// 11.		-   2 a       < e_i.y , T >				< e_i.y , T >		= \sum_j    y[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
	// 12.		-/+ 2 b       < e_i.x , T >				< e_i.x , T >		= \sum_j    x[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
	// 13.		-   2         < x.y , T >				< x.y , T >			= \sum_jk   x[j]*y[k]           < e_j.e_k , T > = e_k.dot( _B * e_j )
	// 14.		+             < x.y , x.y >				< x.y , x.y >		= \sum_jklm x[j]*y[k]*x[l]*y[m] < e_j.e_k , e_l.e_m >
	// 15.		+             < T , T >					< T , T >			=                               < T , T >
	// [Note] For asymmetric products, e_i.e_i=0 so terms [1]-[5] vanish and we end up with bi-quadratic energy
	// [Note] Terms [13]-[15] are constants
	// [Recall] < v.w , T > = < _B * v , w >

	for( Eigen::InnerIterator it(_B,idx) ; it ; ++it )
	{
		// 5.		-   2 a  *b   < e_i.e_i , T >			< e_i.e_i , T >		=                               < e_i.e_i , T > = e_i.dot( _B * e_i )
		if constexpr( Sym ) if( it.row()==it.col() ) Q.coefficient(1,1) -= 2. * it.value();
		// 11.		-   2 a       < e_i.y , T >				< e_i.y , T >		= \sum_j    y[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
		Q.coefficient(1,0) -= 2. * it.value() * y[ it.row() ];
		// 12.		-/+ 2 b       < e_i.x , T >				< e_i.x , T >		= \sum_j    x[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
		if constexpr( Sym ) Q.coefficient(0,1) -= 2. * it.value() * x[ it.row() ];
		else                Q.coefficient(0,1) += 2. * it.value() * x[ it.row() ];
	}

	// The index 
	Hat::Index< Dim > F1 = scalars.functionIndex( idx );
	size_t i1 = idx;

	struct ProductCoefficients
	{
		ProductCoefficients( void ) : xy(0) , ey(0) , xe(0) , ee(0) {}
		ProductCoefficients( size_t idx , const Eigen::VectorXd &x , const Eigen::VectorXd &y , size_t i1 , size_t i2 )
		{
			std::pair< double , double > _x( x[i1] , x[i2] ) , _y( y[i1] , y[i2] ) , _e( idx==i1 ? 1 : 0 , idx==i2 ? 1 : 0 );

			xy = Hat::ProductFunctions< Dim , Sym >::Coefficient( _x , _y , i1 , i2 );
			ey = Hat::ProductFunctions< Dim , Sym >::Coefficient( _e , _y , i1 , i2 );
			xe = Hat::ProductFunctions< Dim , Sym >::Coefficient( _x , _e , i1 , i2 );
			ee = Hat::ProductFunctions< Dim , Sym >::Coefficient( _e , _e , i1 , i2 );
		};
		double xy , ey , xe , ee;
	};

	const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Row > &rows = _pMStencil.rows( F1 );
	for( unsigned int i=0 ; i<rows.size() ; i++ )
	{
		Hat::Index< Dim > F2 = F1 + rows[i].F2;
		size_t i2 = scalars.functionIndex( F2 );
		ProductCoefficients F( idx , x , y , i1 , i2 );

		double Q22=0 , Q21=0 , Q12=0 , Q11a=0 , Q11b=0 , Q20=0 , Q02=0 , Q10=0 , Q01=0;
		const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Entry > &entries = rows[i].entries;
		for( unsigned int j=0 ; j<entries.size() ; j++ )
		{
			Hat::Index< Dim > G1 = F1 + entries[j].G1 , G2 = F1 + entries[j].G2;
			size_t j1 = scalars.functionIndex( G1 ) , j2 = scalars.functionIndex( G2 );

			double s = entries[j].value;
			ProductCoefficients G( idx , x , y , j1 , j2 );
			G.xy *= s , G.ey *= s , G.xe *= s , G.ee *= s;

			if constexpr( Sym )
			{
				if( i1==i2 && i1==j1 && i1==j2 ) // & , &
				{
					Q22 += G.ee;
				}
				if( i1==i2 && ( i1==j1 || i1==j2 ) ) // & , |
				{
					Q21 += G.ey;
					Q12 += G.xe;
				}
				if( i1==i2 ) // & , *
				{
					Q11a += G.xy;
				}
			}
			if( i1==j1 || i1==j2 ) // | , |
			{
				Q20 += G.ey;
				Q02 += G.xe;
				Q11b += G.xe;
			}
			// | , *
			Q10 += G.xy;
			Q01 += G.xy;
		}
		Q.coefficient(2,2) +=     Q22 * F.ee;
		Q.coefficient(2,1) += 2 * Q21 * F.ee;
		Q.coefficient(1,2) += 2 * Q12 * F.ee;
		Q.coefficient(2,0) +=     Q20 * F.ey;
		Q.coefficient(0,2) +=     Q02 * F.xe;
		Q.coefficient(1,1) += 2 * ( Q11a * F.ee + Q11b * F.ey );
		Q.coefficient(1,0) += 2 * Q10 * F.ey;
		Q.coefficient(0,1) += 2 * Q01 * F.xe;
	}
	return Q;
}

template< unsigned int Dim , bool Sym , typename Indexer >
Polynomial::Polynomial1D< 4 > CascadicSystemEnergy< Dim , Sym , Indexer >::quarticFit( const Eigen::VectorXd &x , size_t idx , unsigned int thread ) const
{
	if constexpr( !Sym ) MK_ERROR_OUT( "Quartic fit not supported for asymmetric" );

	Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( static_cast< size_t >(idx) , thread );
	size_t f1 = idx;

	Polynomial::Polynomial1D< 4 > Q;

	// [NOTE] The factor of two is needed because the regularization is applied twice (once for x and once for y=x)
	// E(s) = 2 * ( x + s*e_i )^t * R * ( x + s*e_i )
	//      = 2 * x^t * R * x + 2s x^t * R * e_i + s^2 e_i^t * R * e_i

	// Get the contribution from the regulrizer matrix
	for( Eigen::InnerIterator it(_R,idx) ; it ; ++it )
	{
		if( it.row()==idx ) Q.coefficient(2) += 2. * it.value();
		Q.coefficient(1) += 4. * it.value() * x[ it.row() ];
	}

	// Get the contribution from the regularizer stencil
	if( _sWeight )
	{
		Hat::Index< Dim > F1 = indexer.functionIndex( idx );
		const typename Hat::ScalarFunctions< Dim >::template FullIntegrationStencil< double , 0 >::Row &row = _sStencil.row( F1 );
		for( unsigned int i=0 ; i<row.size() ; i++ )
		{
			size_t f2 = neighbors.data[ std::get<1>( row[i] ) ];
			if( f2!=-1 )
			{
				double value = std::get<2>( row[i] ) * _sWeight;
				if( f1==f2 ) Q.coefficient(2) += 2. * value;
				Q.coefficient(1) += 4. * value * x[f2];
			}
		}
	}

	///////////////////////////////////////
	// More generally:
	// Using "a.b" to denote either the symmetric product or the alternating product of a and b:
	//	 E_{ij}(a,b) = || ( a e_i + x ).( b e_j + y ) - T ||^2
	//	             = || ab e_i.e_j + a e_i.y + b x.e_j + x.y - T ||^2
	// Expanding out we get:
	// [QUARTIC]
	// 1.       =     a^2*b^2 < e_i.e_j , e_i.e_j >
	// [CUBIC]
	// 2.		+   2 a^2*b   < e_i.e_j , e_i.y >		< e_i.e_j , e_i.y >	= \sum_l    y[k]                < e_i.e_j , e_i.e_k >
	// 3.		+/- 2 a  *b^2 < e_i.e_j , e_j.x >		< e_i.e_j , x.e_j >	= \sum_l    x[k]                < e_i.e_j , e_i.e_k >
	// [QUADRATIC]
	// 4.		+   2 a  *b   < e_i.e_j , x.y >			< e_i.e_j , x.y >	= \sum_kl   x[k]*y[l]           < e_i.e_j , e_k.e_l >
	// 5.		-   2 a  *b   < e_i.e_j , T >			< e_i.e_j , T >		=                               < e_i.e_j , T > = e_i.dot( _B * e_j )
	// 6.		+     a^2     < e_i.y , e_i.y >			< e_i.y , e_i.y >	= \sum_kl   y[k]*y[l]           < e_i.e_k , e_i.e_l >
	// 7.		+     b^2     < e_j.x , e_j.x >			< e_i.x , e_i.x >	= \sum_kl   x[k]*x[l]           < e_j.e_k , e_j.e_l >
	// 8.		+/- 2 a  *b   < e_i.y , e_j.x >			< e_i.y , e_i.x >	= \sum_kl   x[k]*y[l]           < e_i.e_k , e_j.e_l >
	// [LINEAR]
	// 9.		+   2 a       < e_i.y , x.y >			< e_i.y , x.y >		= \sum_klm  y[k]*x[l]*y[m]      < e_i.e_k , e_l.e_m >
	// 10.		+/- 2 b       < e_j.x , x.y >			< e_i.x , x.y >		= \sum_klm  x[k]*x[l]*y[m]      < e_j.e_k , e_l.e_m >
	// 11.		-   2 a       < e_i.y , T >				< e_i.y , T >		= \sum_k    y[k]                < e_i.e_k , T > = e_i.dot( _B * e_k )
	// 12.		-/+ 2 b       < e_j.x , T >				< e_j.x , T >		= \sum_k    x[k]                < e_j.e_k , T > = e_j.dot( _B * e_k )
	// [CONSTANT]
	// 13.		-   2         < x.y , T >				< x.y , T >			= \sum_kl   x[k]*y[l]           < e_k.e_l , T > = e_k.dot( _B * e_l )
	// 14.		+             < x.y , x.y >				< x.y , x.y >		= \sum_klmn x[k]*y[l]*x[m]*y[n] < e_k.e_l , e_m.e_n >
	// 15.		+             < T , T >					< T , T >			=                               < T , T >


	///////////////////////////////////////
	// Using "a.b" to denote either the symmetric product or the alternating product of a and b:
	//	 E(a,b) = || ( a e_i + x ).( b e_i + y ) - T ||^2
	//	        = || ab e_i.e_i + a e_i.y + b x.e_i + x.y - T ||^2
	// Expanding out we get:
	// 1.       =     a^2*b^2 < e_i.e_i , e_i.e_i >
	// 2.		+   2 a^2*b   < e_i.e_i , e_i.y >		< e_i.e_i , e_i.y >	= \sum_j    y[j]                < e_i.e_i , e_i.e_j >
	// 3.		+/- 2 a  *b^2 < e_i.e_i , e_i.x >		< e_i.e_i , e_i.x >	= \sum_j    x[j]                < e_i.e_i , e_i.e_j >
	// 4.		+   2 a  *b   < e_i.e_i , x.y >			< e_i.e_i , x.y >	= \sum_jk   x[j]*y[k]           < e_i.e_i , e_j.e_k >
	// 5.		-   2 a  *b   < e_i.e_i , T >			< e_i.e_i , T >		=                               < e_i.e_i , T > = e_i.dot( _B * e_i )
	// 6.		+     a^2     < e_i.y , e_i.y >			< e_i.y , e_i.y >	= \sum_jk   y[j]*y[k]           < e_i.e_j , e_i.e_k >
	// 7.		+     b^2     < e_i.x , e_i.x >			< e_i.x , e_i.x >	= \sum_jk   x[j]*x[k]           < e_i.e_j , e_i.e_k >
	// 8.		+/- 2 a  *b   < e_i.y , e_i.x >			< e_i.y , e_i.x >	= \sum_jk   x[j]*y[k]           < e_i.e_j , e_i.e_k >
	// 9.		+   2 a       < e_i.y , x.y >			< e_i.y , x.y >		= \sum_jkl  y[j]*x[k]*y[l]      < e_i.e_j , e_k.e_l >
	// 10.		+/- 2 b       < e_i.x , x.y >			< e_i.x , x.y >		= \sum_jkl  x[j]*x[k]*y[l]      < e_i.e_j , e_k.e_l >
	// 11.		-   2 a       < e_i.y , T >				< e_i.y , T >		= \sum_j    y[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
	// 12.		-/+ 2 b       < e_i.x , T >				< e_i.x , T >		= \sum_j    x[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
	// 13.		-   2         < x.y , T >				< x.y , T >			= \sum_jk   x[j]*y[k]           < e_j.e_k , T > = e_k.dot( _B * e_j )
	// 14.		+             < x.y , x.y >				< x.y , x.y >		= \sum_jklm x[j]*y[k]*x[l]*y[m] < e_j.e_k , e_l.e_m >
	// 15.		+             < T , T >					< T , T >			=                               < T , T >
	// [Note] For asymmetric products, e_i.e_i=0 so terms [1]-[5] vanish and we end up with bi-quadratic energy
	// [Note] Terms [13]-[15] are constants
	// [Recall] < v.w , T > = < _B * v , w >

	for( Eigen::InnerIterator it(_B,idx) ; it ; ++it )
	{
		// 5.		-   2 a  *b   < e_i.e_i , T >			< e_i.e_i , T >		=                               < e_i.e_i , T > = e_i.dot( _B * e_i )
 		if constexpr( Sym ) if( it.row()==it.col() ) Q.coefficient(2) -= 2. * it.value();
		// 11.		-   2 a       < e_i.y , T >				< e_i.y , T >		= \sum_j    y[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
		Q.coefficient(1) -= 2. * it.value() * x[ it.row() ];
		// 12.		-/+ 2 b       < e_i.x , T >				< e_i.x , T >		= \sum_j    x[j]                < e_i.e_j , T > = e_j.dot( _B * e_i )
		if constexpr( Sym ) Q.coefficient(1) -= 2. * it.value() * x[ it.row() ];
		else                Q.coefficient(1) += 2. * it.value() * x[ it.row() ];
	}

	// The index 
	Hat::Index< Dim > F1 = indexer.functionIndex( idx );

	struct ProductCoefficients
	{
		ProductCoefficients( const double *x , size_t f1 , size_t f2 )
		{
			std::pair< double , double > _x( x[f1] , x[f2] ) , _e( f1 ? 0 : 1 , f2 ? 0 : 1 );

			xx = _x.first * _x.second;
			ex = ( _e.first * _x.second  + _e.second * _x.first )/2.;
			ee = _e.first * _e.second;
		};
		double xx , ex , ee;
	};
	struct GlobalProductCoefficients
	{
		GlobalProductCoefficients( const Eigen::VectorXd &x , size_t f , size_t f1 , size_t f2 )
		{
			std::pair< double , double > _x( x[f1] , x[f2] ) , _e( f1!=f ? 0 : 1 , f2!=f ? 0 : 1 );

			xx = _x.first * _x.second;
			ex = ( _e.first * _x.second  + _e.second * _x.first )/2.;
			ee = _e.first * _e.second;
		};
		double xx , ex , ee;
	};

	double Q4=0 , Q3=0 , Q2=0 , Q1=0;

	const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Row > &rows = __pMStencil.rows( F1 );

	{
		GlobalProductCoefficients F( x , f1 , f1 , f1 );

		const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Entry > &entries = rows[0].entries;
		for( unsigned int j=0 ; j<entries.size() ; j++ )
		{
			size_t g1 = neighbors.data[ entries[j]._g1 ] , g2 = neighbors.data[ entries[j]._g2 ];
			if( g1!=-1 && g2!=-1 )
			{
				GlobalProductCoefficients G( x , f1 , g1 , g2 );

				Q4 += G.ee * entries[j].value;
				Q3 += G.ex * entries[j].value;
				Q2 += G.xx * entries[j].value;
			}
		}
		Q4 *=     F.ee;
		Q3 *= 4 * F.ee;
		Q2 *= 2 * F.ee;
	}

	for( unsigned int i=0 ; i<rows.size() ; i++ )
	{
		size_t f2 = neighbors.data[ rows[i]._f2 ];
		if( f2!=-1 )
		{
			GlobalProductCoefficients F( x , f1 , f1 , f2 );

			double _Q2=0 , _Q1=0;
			const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Entry > &entries = rows[i].entries;
			// (j1,j2) = (0,0)
			{
				_Q2 += 2 * x[f1] * entries[0].value;
				_Q1 += x[f1] * x[f1] * entries[0].value;
			}
			// (j1,j2) = (0,*)
			for( unsigned int j=1 ; j<rows[i].end0_ ; j++ )
			{
				size_t g2 = neighbors.data[ entries[j]._g2 ];
				if( g2!=-1 )
				{
					double __x = x[g2] * entries[j].value;
					_Q2 += __x;
					_Q1 += x[f1] * __x;
				}
			}
			// (j1,j2) = (*,0)
			for( unsigned int j=rows[i].end0_ ; j<rows[i].end_0; j++ )
			{
				size_t g1 = neighbors.data[ entries[j]._g1 ];
				if( g1!=-1 )
				{
					double __x = x[g1] * entries[j].value;
					_Q2 += __x;
					_Q1 += x[f1] * __x;
				}
			}
			// (j1,j2) = (*,*)
			for( unsigned int j=rows[i].end_0 ; j<entries.size() ; j++ )
			{
				size_t g1 = neighbors.data[ entries[j]._g1 ] , g2 = neighbors.data[ entries[j]._g2 ];
				if( g1!=-1 && g2!=-1 ) _Q1 += x[g1] * x[g2] * entries[j].value;
			}
			Q2 += 2 * _Q2 * F.ex;
			Q1 += 4 * _Q1 * F.ex;
		}
	}
	Q.coefficient(4) += Q4;
	Q.coefficient(3) += Q3;
	Q.coefficient(2) += Q2;
	Q.coefficient(1) += Q1;

	return Q;
}

template< unsigned int Dim , bool Sym , typename Indexer >
double CascadicSystemEnergy< Dim , Sym , Indexer >::squareNorm( const Eigen::VectorXd &x ) const
{
	if constexpr( !Sym ) MK_ERROR_OUT( "Square-norm not supported for asymmetric" );

	double n2 = 0;

	ThreadPool::ParallelFor
		(
			0 , indexer.numFunctions() ,
			[&]( unsigned int thread , size_t f1 )
			{
				Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , thread );
				Hat::Index< Dim > Off;
				for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;

				Hat::Index< Dim > F1 = indexer.functionIndex( f1 );

				const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Row > &rows = __pMStencil.rows( F1 );

				for( unsigned int i=0 ; i<rows.size() ; i++ )
				{
					if( rows[i]._f2<Window::IsotropicIndex< Dim , 3 >::template I< 1 >() ) break;
					size_t f2 = neighbors.data[ rows[i]._f2 ];
					if( f2!=-1 )
					{
						double _n2 = 0;
						const std::vector< typename Hat::ProductFunctions< Dim , Sym >::template FullIntegrationStencil< double >::Entry > &entries = rows[i].entries;

						for( unsigned int j=0 ; j<entries.size() ; j++ )
						{
							size_t g1 = neighbors.data[ entries[j]._g1 ] , g2 = neighbors.data[ entries[j]._g2 ];
							if( g1!=-1 && g2!=-1 ) _n2 += x[g1] * x[g2] * entries[j].value;
						}
						Atomic< double >::Add( n2 , _n2 * x[f1] * x[f2] );
					}
				}
			}
		);

	return n2;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::QuadraticApproximation CascadicSystemEnergy< Dim , Sym , Indexer >::quadraticApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	//tex:
	// Putting all this together we get:
	// \begin{align*}
	// \frac{dE}{dx}       &= 2\left( \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y}\cdot x \mp \mathbf{Y}^\top\cdot\mathbf{M}\cdot \tau + \mathbf{R}\cdot x\right)\\
	// \frac{dE}{dy}       &= 2\left( \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X}\cdot y  -  \mathbf{X}^\top\cdot\mathbf{M}\cdot \tau + \mathbf{R}\cdot y\right)\\
	// \frac{d^2E}{dx^2}   &= 2\left( \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{Y} + \mathbf{R}\right)\\
	// \frac{d^2E}{dy^2}   &= 2\left( \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{X} + \mathbf{R}\right)\\
	// \frac{d^2E}{dy\,dx} &= 2\left( \pm \mathbf{Y}^\top\cdot\mathbf{M}\cdot\mathbf{X} \pm \mathbf{A}_{\mathbf{M}\cdot x\times y} \mp \mathbf{A}_{\mathbf{M}\cdot\tau}\right)\\
	// \frac{d^2E}{dx\,dy} &= 2\left( \pm \mathbf{X}^\top\cdot\mathbf{M}\cdot\mathbf{Y}  +  \mathbf{A}_{\mathbf{M}\cdot x\times y}  -  \mathbf{A}_{\mathbf{M}\cdot\tau}\right)
	// \end{align*}f
	typename Energy< Dim , Sym , Indexer >::QuadraticApproximation qa;
	Eigen::SparseMatrix< double > Y = products.product(y);
	Eigen::SparseMatrix< double > MY = products.mass() * Y;

	qa.c = Energy< Dim , Sym , Indexer >::operator()( x , y );
	if constexpr( Sym )	qa.l = 2 * ( Y.transpose() * ( MY * x - products.toVector( _B ) ) + _R * x );
	else                qa.l = 2 * ( Y.transpose() * ( MY * x + products.toVector( _B ) ) + _R * x );
	if( _sWeight ) qa.l += 2 * scalars( indexer , _sStencil , x ) * _sWeight;
	qa.q = Y.transpose() * MY + _R;
	if( _sWeight ) qa.q += scalars.systemMatrix( indexer , _sStencil ) * _sWeight;
	return qa;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::QuadraticApproximation CascadicSystemEnergy< Dim , Sym , Indexer >::quadraticApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
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
	typename Energy< Dim , Sym , Indexer >::QuadraticApproximation qa;
	Eigen::SparseMatrix< double > X = products.product(x);
	Eigen::SparseMatrix< double > MX = products.mass() * X;

	qa.c = Energy< Dim , Sym , Indexer >::operator()( x , y );
	qa.l = 2 * ( X.transpose() * ( MX * y - products.toVector( _B ) ) + _R * y );
	if( _sWeight ) qa.l += 2 * scalars( indexer , _sStencil , y ) * _sWeight;
	qa.q = X.transpose() * MX + _R;
	if( _sWeight ) qa.q += scalars.systemMatrix( indexer , _sStencil ) * _sWeight;
	return qa;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::QuadraticApproximation CascadicSystemEnergy< Dim , Sym , Indexer >::quadraticApproximation( const Eigen::VectorXd &x ) const
{
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
	if constexpr( !Sym ) MK_ERROR_OUT( "Quadratic approximation not supported for asymmetric" );

	Eigen::SparseMatrix< double > X = products.product(x);
	Eigen::SparseMatrix< double > MX = products.mass() * X;

	typename Energy< Dim , Sym , Indexer >::QuadraticApproximation qa;
	qa.c = Energy< Dim , Sym , Indexer >::operator()(x);
	qa.l = 4 * ( X.transpose() * ( MX * x - products.toVector( _B ) ) + _R * x );
	if( _sWeight ) qa.l += 4 * scalars( indexer , _sStencil , x ) * _sWeight;
	qa.q = 2 * ( 2 * X.transpose() * MX + products.toMatrix( products.valueDual( products.product(x,x) ) ) - _B + _R );
	if( _sWeight ) qa.q += 2 * scalars.systemMatrix( indexer , _sStencil ) * _sWeight;

	return qa;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::LinearApproximation CascadicSystemEnergy< Dim , Sym , Indexer >::linearApproximation1( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
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

	typename Energy< Dim , Sym , Indexer >::LinearApproximation la;
	la.c = Energy< Dim , Sym , Indexer >::operator()( x , y );
	if constexpr( Sym ) la.l = 2 * ( products.productTranspose( y , products.valueDual( products.product(y,x) ) - products.toVector( _B ) ) + _R * x );
	else                la.l = 2 * ( products.productTranspose( y , products.valueDual( products.product(y,x) ) + products.toVector( _B ) ) + _R * x );
	if( _sWeight ) la.l += 2 * scalars( indexer , _sStencil , x ) * _sWeight;

	return la;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::LinearApproximation CascadicSystemEnergy< Dim , Sym , Indexer >::linearApproximation2( const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
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

	typename Energy< Dim , Sym , Indexer >::LinearApproximation la;
	la.c = Energy< Dim , Sym , Indexer >::operator()( x , y );
	la.l = 2 * ( products.productTranspose( x , products.valueDual( products.product(x,y) ) - products.toVector( _B ) ) + _R * y );
	if( _sWeight ) la.l += 2 * scalars( indexer , _sStencil , y ) * _sWeight;
	return la;
}

template< unsigned int Dim , bool Sym , typename Indexer >
typename Energy< Dim , Sym , Indexer >::LinearApproximation CascadicSystemEnergy< Dim , Sym , Indexer >::linearApproximation( const Eigen::VectorXd &x ) const
{
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
	if constexpr( !Sym ) MK_ERROR_OUT( "Quadratic approximation not supported for asymmetric" );

	typename Energy< Dim , Sym , Indexer >::LinearApproximation la;
	la.c = Energy< Dim , Sym , Indexer >::operator()(x);
	la.l = 4 * ( products.productTranspose( x , products.valueDual( products.product(x,x) ) - products.toVector(_B) ) + _R * x );
	if( _sWeight ) la.l += 4 * scalars( indexer , _sStencil , x ) * _sWeight;

	return la;
}