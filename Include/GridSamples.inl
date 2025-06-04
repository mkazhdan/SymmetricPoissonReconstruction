/*
Copyright (c) 2025, Michael Kazhdan, Sing-Chun Lee, Marc Alexa, and Maximilian Kohlbrenner
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

/////////////
// BSpline //
/////////////
template< unsigned int Degree >
void BSpline< Degree >::Evaluate( double x , double values[Degree+1] )
{
	if constexpr( Degree==0 ) values[0] = 1.;
	else
	{
		const double Scale = 1./Degree;
		BSpline< Degree-1 >::Evaluate( x , values+1 );
		values[0] = values[1] * (1.-x) * Scale;
		for( unsigned int i=1 ; i<Degree ; i++ )
		{
			double x1 = (x-i+Degree) , x2 = (-x+i+1);
			values[i] = ( values[i]*x1 + values[i+1]*x2 ) * Scale;
		}
		values[Degree] *= x * Scale;
	}
}

void BSpline< (unsigned int)-1 >::Evaluate( unsigned int degree , double x , double values[/*degree+1*/] )
{
	if( degree==0 ) values[0] = 1.;
	else
	{
		const double Scale = 1./degree;
		Evaluate( degree-1 , x , values+1 );
		values[0] = values[1] * (1.-x) * Scale;
		for( unsigned int i=1 ; i<degree ; i++ )
		{
			double x1 = (x-i+degree) , x2 = (-x+i+1);
			values[i] = ( values[i]*x1 + values[i+1]*x2 ) * Scale;
		}
		values[degree] *= x * Scale;
	}
}

/////////////////
// DensityInfo //
/////////////////
template< unsigned int Dim >
DensityInfo< Dim >::DensityInfo( void ) : weightSum(0) {}

template< unsigned int Dim >
DensityInfo< Dim >::DensityInfo( Point< double , Dim > p , double weight ) : weightSum(weight) , positionSum(p*weight) {}

template< unsigned int Dim >
void DensityInfo< Dim >::Add( const DensityInfo &ni ){ weightSum += ni.weightSum , positionSum += ni.positionSum; }

template< unsigned int Dim >
void DensityInfo< Dim >::Scale( double s ){ weightSum *= s , positionSum *= s; }

template< unsigned int Dim >
double DensityInfo< Dim >::noise( unsigned int coDim ) const { return 1.; }

template< unsigned int Dim >
double DensityInfo< Dim >::measure( void ) const { return 1./weightSum; }

/////////////////////////
// DensityAndNoiseInfo //
/////////////////////////
template< unsigned int Dim >
DensityAndNoiseInfo< Dim >::DensityAndNoiseInfo( void ) : weightSum(0) {}

template< unsigned int Dim >
DensityAndNoiseInfo< Dim >::DensityAndNoiseInfo( Point< double , Dim > p , double weight ) : weightSum(weight) , positionSum(p*weight){ for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) covarianceSum(i,j) = p[i] * p[j] * weight; }

template< unsigned int Dim >
void DensityAndNoiseInfo< Dim >::Add  ( const DensityAndNoiseInfo &ni ){ weightSum += ni.weightSum , positionSum += ni.positionSum , covarianceSum += ni.covarianceSum; }

template< unsigned int Dim >
void DensityAndNoiseInfo< Dim >::Scale( double s ){ weightSum *= s , positionSum *= s , covarianceSum *= s; }

template< unsigned int Dim >
double DensityAndNoiseInfo< Dim >::noise( unsigned int coDim ) const
{
	// Recall that if \sum_i w_i = 1 and c = \sum_i w_i * p_i
	//		C = \sum_i w_i * ( p_i - c ) * ( p_i - c )^t
	//		  = \sum_i w_i * ( p_i * p_i^t - p_i * c^t - c * p_i^t + c * c^t )
	//		  = \sum_i w_i * p_i * p_i^t - \sum_i w_i( p_i * c^t + c * p_i^t ) + ( \sum_i w_i ) * c * c^t
	//		  = \sum_i w_i * p_i * p_i^t - 2 * c * c^t + c * c^t
	//		  = \sum_i w_i * p_i * p_i^t - c * c^t
	Eigen::Matrix< double , Dim , Dim > C;

	for( unsigned int i=0 ; i<Dim ; i++ ) for( unsigned int j=0 ; j<Dim ; j++ ) C(i,j) = covarianceSum(i,j)/weightSum - positionSum[i]/weightSum * positionSum[j]/weightSum;
	Eigen::SelfAdjointEigenSolver< Eigen::Matrix< double , Dim , Dim > > es( C );
	double sum = 0 , lowSum = 0;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		sum += es.eigenvalues()[d];
		if( d<coDim ) lowSum += es.eigenvalues()[d];
	}
	return ( lowSum * Dim ) /( sum * coDim );
}

template< unsigned int Dim >
double DensityAndNoiseInfo< Dim >::measure( void ) const { return 1./weightSum; }

//////////////
// Splatter //
//////////////
template< unsigned int Dim >
Splatter< Dim >::Splatter( unsigned int depth , unsigned int bSplineDegree , bool unitKernel ) : _res(1<<depth) , _scalars(1<<depth) , _bSplineDegree(bSplineDegree)
{
	_values = new double[ Dim * ( _bSplineDegree+1 ) ];
	_bSplineStart = -(int)(_bSplineDegree>>1);
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		_range.first [d] = _bSplineStart;
		_range.second[d] = _bSplineStart+_bSplineDegree+1;
	}
	_kernelScale = unitKernel ? ( 1<<(depth*Dim) ) : 1.;
}
template< unsigned int Dim >
Splatter< Dim >::~Splatter( void ){ delete _values; }

template< unsigned int Dim >
template< typename DataType >
void Splatter< Dim >::operator()( DataType *g , std::pair< Point< double , Dim > , DataType > sample )
{
	Hat::Index< Dim > e;
	sample.first *= _res;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		e[d] = (int)floor( sample.first[d] );
		BSpline<>::Evaluate( _bSplineDegree , sample.first[d]-e[d] , _values+d*(_bSplineDegree+1) );
	}

	auto f = [&]( Hat::Index< Dim > _f )
		{
			Hat::Index< Dim > f = _f + e;
			double scale = _kernelScale;
			bool inRange = true;
			for( unsigned int d=0 ; d<Dim ; d++ )
				if( f[d]>=0 && f[d]<=(int)_res ) scale *= _values[ d * ( _bSplineDegree+1) + _f[d] - _bSplineStart ];
				else inRange = false;
			if( inRange ) g[ _scalars.functionIndex( f ) ] += scale * sample.second;
		};
	_range.process(f);
}

template< unsigned int Dim >
template< typename DataType , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int idx ) */ >
void Splatter< Dim >::SplatSamples( unsigned int bSplineDegree , RegularGrid< Dim , DataType > &g , SampleFunctor F , size_t sampleNum , bool unitKernel )
{
	unsigned int res = g.res(0)-1;
	for( unsigned int d=1 ; d<Dim ; d++ ) if( g.res(d)-1!=res ) MK_ERROR_OUT( "Not a cubical grid" );
	unsigned int depth = 0;
	while( (1u<<depth)<res ) depth++;
	if( res!=(1<<depth) ) MK_ERROR_OUT( "Resolution is not a power of two: " , res );

	Splatter splatter( depth , bSplineDegree , unitKernel );
	for( unsigned int s=0 ; s<sampleNum ; s++ ) splatter( g() , F(s) );
}

template< unsigned int Dim >
template< typename DataType , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int idx ) */ >
void Splatter< Dim >::SplatSamples_parallel( unsigned int bSplineDegree , RegularGrid< Dim , DataType > &g , SampleFunctor F , size_t sampleNum , DataType zeroValue , bool unitKernel )
{
	unsigned int res = g.res(0)-1;
	for( unsigned int d=1 ; d<Dim ; d++ ) if( g.res(d)-1!=res ) MK_ERROR_OUT( "Not a cubical grid" );
	unsigned int depth = 0;
	while( (1u<<depth)<res ) depth++;
	if( res!=(1<<depth) ) MK_ERROR_OUT( "Resolution is not a power of two: " , res );

	std::vector< RegularGrid< Dim , DataType > > gs( ThreadPool::NumThreads() );
	std::vector< Splatter > splatters;
	splatters.reserve( ThreadPool::NumThreads() );
	for( int t=0 ; t<ThreadPool::NumThreads() ; t++ )
	{
		gs[t].resize( g.res() );
		splatters.emplace_back( depth , bSplineDegree , unitKernel );
	}

	ThreadPool::ParallelFor( 0 , sampleNum , [&]( unsigned int t , size_t s ){ splatters[t]( gs[t]() , F(s) ); } );
	ThreadPool::ParallelFor( 0 , g.resolution() , [&]( unsigned int , size_t i ){ for( unsigned int t=0 ; t<gs.size() ; t++ ) g[i] += gs[t][i]; } );
}

template< unsigned int Dim >
template< typename DataType , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int idx ) */ >
RegularGrid< Dim , DataType > Splatter< Dim >::SplatSamples( unsigned int bSplineDegree , unsigned int depth , SampleFunctor F , size_t sampleNum , DataType zeroValue , bool unitKernel )
{
	RegularGrid< Dim , DataType > g;
	{
		unsigned int res[Dim];
		for( unsigned int d=0 ; d<Dim ; d++ ) res[d] = (1<<depth) + 1;
		g.resize( res );
		for( size_t i=0 ; i<g.resolution() ; i++ ) g[i] = zeroValue;
	}
	SplatSamples( bSplineDegree , g , F , sampleNum , unitKernel );
	return g;
}

/////////////////////
// SingleEstimator //
/////////////////////
template< unsigned int Dim , unsigned int CoDim >
SingleEstimator< Dim , CoDim >::SingleEstimator( unsigned int depth ) : _depth(depth)
{
	unsigned int res[Dim];
	for( unsigned int dd=0 ; dd<Dim ; dd++ ) res[dd] = (1<<depth) + 1;
	RegularGrid< Dim , DensityAndNoiseInfo< Dim > >::resize( res );
}

template< unsigned int Dim , unsigned int CoDim >
unsigned int SingleEstimator< Dim , CoDim >::depth( void ) const { return _depth; }

template< unsigned int Dim , unsigned int CoDim >
DensityAndNoiseInfo< Dim > SingleEstimator< Dim , CoDim >::operator()( Point< double , Dim > p , unsigned int ) const { return RegularGrid< Dim , DensityAndNoiseInfo< Dim > >::operator()( p*(1<<_depth) ); }

template< unsigned int Dim , unsigned int CoDim >
double SingleEstimator< Dim , CoDim >::measure( Point< double , Dim > p , unsigned int thread ) const { return this->operator()( p , thread ).measure(); }

template< unsigned int Dim , unsigned int CoDim >
double SingleEstimator< Dim , CoDim >::noise( Point< double , Dim > p , unsigned int thread ) const { return this->operator()( p , thread ).noise( CoDim ); }

template< unsigned int Dim , unsigned int CoDim >
double SingleEstimator< Dim , CoDim >::depth( Point< double , Dim > p , unsigned int ) const { return _depth; }

template< unsigned int Dim , unsigned int CoDim >
double SingleEstimator< Dim , CoDim >::samplesPerCell( Point< double , Dim > p , unsigned int thread ) const { return this->operator()( p , thread ).weightSum / ( 1<<(_depth*(Dim-CoDim) ) ); }

template< unsigned int Dim , unsigned int CoDim >
template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( unsigned int idx ) */ >
SingleEstimator< Dim , CoDim > SingleEstimator< Dim , CoDim >::Get( unsigned int kernelRadius , unsigned int depth , SampleFunctor F , size_t sampleNum )
{
	SingleEstimator estimator( depth );
	Set( estimator , kernelRadius , F , sampleNum );
	return estimator;
}

template< unsigned int Dim , unsigned int CoDim >
template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( unsigned int idx ) */ >
void SingleEstimator< Dim , CoDim >::Set( SingleEstimator< Dim , CoDim > &estimator , unsigned int kernelRadius , SampleFunctor F , size_t sampleNum )
{
	double scale;
	{
		unsigned int bSplineDegree = 2*kernelRadius + 1;
		double *_values = new double[bSplineDegree+1];
		BSpline<>::Evaluate( bSplineDegree , 0.5 , _values );
		scale = pow( 1./_values[kernelRadius+1] , CoDim );
		delete[] _values;
	}
	auto _F = [&]( size_t idx ){ return std::make_pair( F(idx) , DensityAndNoiseInfo< Dim >( F(idx) , scale / pow( pow( 0.5 , estimator.depth() ) , Dim - CoDim ) ) ); };
	Splatter< Dim >::template SplatSamples_parallel< DensityAndNoiseInfo< Dim > >( 2*kernelRadius+1 , estimator , _F , sampleNum , DensityAndNoiseInfo< Dim >() , false );
}

////////////////////
// MultiEstimator //
////////////////////
template< unsigned int Dim , unsigned int CoDim >
MultiEstimator< Dim , CoDim >::MultiEstimator( unsigned int depth , double samplesPerCell ) : _samplesPerCell(samplesPerCell)
{
	_info.reserve( depth+1 );
	for( unsigned int d=0 ; d<=depth ; d++ ) _info.emplace_back( d );
}

template< unsigned int Dim , unsigned int CoDim >
double MultiEstimator< Dim , CoDim >::depth( Point< double , Dim > p , unsigned int thread ) const
{
	// Find the finest depth at which the number of samples per node exceeds _samplesPerCell
	unsigned int d1 = (unsigned int)_info.size()-1 , d2 = (unsigned int)_info.size();
	while( d2 && _info[d1].samplesPerCell( p , thread )<_samplesPerCell ) d1-- , d2--;
	if( d2==0 || d2==_info.size() ) // Extremal case
	{
		unsigned int d;
		if     ( d2==0 ) d=0;
		else if( d2==_info.size() ) d = (unsigned int)_info.size()-1;
		double samplesPerCell = _info[d].samplesPerCell( p , thread );
		// Generically the assumption is that increase the depth by should decrease the number of samples per cell by a factor of 2^{Dim-CoDim}
		// Solve for the depth such that:
		//		_samplesPerCell = samplesPerCell / 2^{depth-d}
		//		2^{depth-d} = samplesPerCell/_samplesPerCell
		//		depth - d = log( samplesPerCell/_samplesPerCell ) / log(2)
		//		depth = log( samplesPerCell/_samplesPerCell ) / log(2) + d
		return d + log( samplesPerCell/_samplesPerCell ) / log(2.);
	}
	else // Interior case
	{
		// Here we assume that the change in the samples per cell is simply samplesPerCell[d2]/samplesPerCell[d1]
		// Solve for the depth such that:
		//		_samplesPerCell = samplesPerCell[d1] * pow( samplesPerCell[d2]/samplesPerCell[d1] , d-d1 )
		//		pow( samplesPerCell[d2]/samplesPerCell[d1] , d-d1 ) = _samplesPerCell / samplesPerCell[d1]
		//		d-d1 = log( _samplesPerCell / samplesPerCell[d1] ) / log( samplesPerCell[d2]/samplesPerCell[d1] )
		//		d = log( _samplesPerCell / samplesPerCell[d1] ) / log( samplesPerCell[d2]/samplesPerCell[d1] ) + d1
		double samplesPerCell1 = _info[d1].samplesPerCell( p , thread ) , samplesPerCell2 = _info[d2].samplesPerCell( p , thread );
		if( !samplesPerCell2 ) return d1;
		else return d1 + log( _samplesPerCell/samplesPerCell1 ) / log( samplesPerCell2/samplesPerCell1 );
	}
}

template< unsigned int Dim , unsigned int CoDim >
DensityAndNoiseInfo< Dim > MultiEstimator< Dim , CoDim >::operator()( Point< double , Dim > p , unsigned int thread ) const
{
	double depth = this->depth( p , thread );
	if( depth>=_info.size()-1 ) return _info.back()( p , thread );
	else if( depth<0 ) return _info[0]( p , thread );
	else
	{
		int d1 = (int)floor(depth) , d2 = (int)floor(depth)+1;
		double d = depth-d1;
		return _info[d1]( p , thread ) * (1.-d) + _info[d2]( p , thread ) * d;
	}
}

template< unsigned int Dim , unsigned int CoDim >
double MultiEstimator< Dim , CoDim >::measure( Point< double , Dim > p , unsigned int thread ) const { return this->operator()( p , thread ).measure(); }

template< unsigned int Dim , unsigned int CoDim >
double MultiEstimator< Dim , CoDim >::  noise( Point< double , Dim > p , unsigned int thread ) const { return this->operator()( p , thread ).noise( CoDim ); }

template< unsigned int Dim , unsigned int CoDim >
double MultiEstimator< Dim , CoDim >::samplesPerCell( Point< double , Dim > p , unsigned int thread ) const
{
	// If at depth d, the cell has _samplesPerCell samples per cell then,
	// assuming that the number samples per cell decreases by a factor of 2^{Dim-CoDim} per level,
	// the number of samples per cell at depth _depth is _samplesPerCell * pow( 0.5 , (Dim-CoDim) * (_depth-d)
	return _samplesPerCell * pow( 0.5 , (Dim-CoDim) * ( (int)_info.size()-1 - depth( p , thread ) ) );
}

template< unsigned int Dim , unsigned int CoDim >
template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( unsigned int idx ) */ >
MultiEstimator< Dim , CoDim > MultiEstimator< Dim , CoDim >::Get( unsigned int kernelRadius , unsigned int depth , SampleFunctor F , size_t sampleNum , double samplesPerCell )
{
	MultiEstimator estimator( depth , samplesPerCell );
	for( unsigned int d=0 ; d<=depth ; d++ ) SingleEstimator< Dim , CoDim >::Set( estimator._info[d] , kernelRadius , F , sampleNum );
	return estimator;
}

//////////////////
// TreeSplatter //
//////////////////

template< unsigned int Dim , typename DataType >
TreeSplatter< Dim , DataType >::TreeSplatter( BinaryStream &stream )
{
	_root.read( stream , nullptr );
	_maxDepth = _root.maxDepth()-1;
	_spaceRoot = _root.children;
	_nKeys.resize( ThreadPool::NumThreads() );
	for( unsigned int i=0 ; i<_nKeys.size() ; i++ ) _nKeys[i].set( _maxDepth );
}

template< unsigned int Dim , typename DataType >
TreeSplatter< Dim , DataType >::TreeSplatter( unsigned int maxDepth ) : _maxDepth( maxDepth )
{
	_root.template initChildren< false >( nullptr );
	_spaceRoot = _root.children;
	_nKeys.resize( ThreadPool::NumThreads() );
	for( unsigned int i=0 ; i<_nKeys.size() ; i++ ) _nKeys[i].set( maxDepth+1 );
}

template< unsigned int Dim , typename DataType >
template< unsigned int KernelRadius , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int idx ) > */ , typename DepthWeightFunctor /* = std::function< double ( unsigned int depth ) > */ >
void TreeSplatter< Dim , DataType >::addSamples( SampleFunctor && Sample , const OrderedSampler< Dim > &orderedSampler , DepthWeightFunctor && DepthWeight , bool mergeSamples )
{
	static_assert( std::is_convertible_v< SampleFunctor , std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int ) > > , "[ERROR] SampleFunctor poorly formed" );
	static_assert( std::is_convertible_v< DepthWeightFunctor , std::function< double ( unsigned int ) > > , "[ERROR] DepthWeightFunctor poorly formed" );

	static const unsigned int BSplineDegree = 2*KernelRadius + 1;

	_InsertionData< KernelRadius > iData( _maxDepth );

	unsigned int res = 1<<_maxDepth;
	for( size_t i=0 ; i<orderedSampler.size() ; i++ )
	{
		Hat::Index< Dim > E = orderedSampler[i].first;
		const std::vector< size_t > &subSampleIndices = orderedSampler[i].second;
		Point< double , Dim > p;
		for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d]+0.5 ) / res;
		Node * n = _spaceRoot->template insertPoint< false >( p , _maxDepth , nullptr );
		iData.nKey.template getNeighbors< true , false >( n , nullptr );

		if( mergeSamples )
		{
			Point< double , Dim > p;
			DataType data{};
			for( size_t j=0 ; j<subSampleIndices.size() ; j++ ) p += Sample( subSampleIndices[j] ).first , data += Sample( subSampleIndices[j] ).second;
			p /= static_cast< double >( subSampleIndices.size() );
			_addSample< KernelRadius >( iData , p , data , std::forward< DepthWeightFunctor >( DepthWeight ) );
		}
		else for( unsigned int j=0 ; j<subSampleIndices.size() ; j++ ) _addSample< KernelRadius >( iData , Sample( subSampleIndices[j] ).first , Sample( subSampleIndices[j] ).second , std::forward< DepthWeightFunctor >( DepthWeight ) );
	}
}

template< unsigned int Dim , typename DataType >
template< unsigned int KernelRadius , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int idx ) > */ , typename SampleDepthFunctor /* = std::function< double ( unsigned int idx ) > */ , typename DepthWeightFunctor /* = std::function< double ( unsigned int depth ) > */ >
void TreeSplatter< Dim , DataType >::addSamples( SampleFunctor && Sample , const OrderedSampler< Dim > &orderedSampler , SampleDepthFunctor && SampleDepth , DepthWeightFunctor && DepthWeight , bool mergeSamples )
{
	static_assert( std::is_convertible_v< SampleFunctor , std::function< std::pair< Point< double , Dim > , DataType > ( unsigned int ) > > , "[ERROR] SampleFunctor poorly formed" );
	static_assert( std::is_convertible_v< SampleDepthFunctor , std::function< double ( unsigned int ) > > , "[ERROR] SampleDepthFunctor poorly formed" );
	static_assert( std::is_convertible_v< DepthWeightFunctor , std::function< double ( unsigned int ) > > , "[ERROR] DepthWeightFunctor poorly formed" );

	static const unsigned int BSplineDegree = 2*KernelRadius + 1;

	_InsertionData< KernelRadius > iData( _maxDepth );

	unsigned int res = 1<<_maxDepth;

	for( size_t i=0 ; i<orderedSampler.size() ; i++ )
	{
		Hat::Index< Dim > E = orderedSampler[i].first;
		const std::vector< size_t > &subSampleIndices = orderedSampler[i].second;
		Point< double , Dim > p;
		for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d]+0.5 ) / res;
		Node *n = _spaceRoot->template insertPoint< false >( p , _maxDepth , nullptr );
		iData.nKey.template getNeighbors< true , false >( n , nullptr );
		if( mergeSamples )
		{
			double depth = 0;
			DataType data{};
			Point< double , Dim > p;
			for( unsigned int j=0 ; j<subSampleIndices.size() ; j++ )
			{
				p += Sample( subSampleIndices[j] ).first;
				depth += SampleDepth( subSampleIndices[j] ) + 1.;
				data += Sample( subSampleIndices[j] ).second;
			}
			p /= static_cast< double >( subSampleIndices.size() );
			depth /= static_cast< double >( subSampleIndices.size() );
			_addSample< KernelRadius >( iData , depth + 1. , p , data , std::forward< DepthWeightFunctor >( DepthWeight ) );
		}
		else
			for( unsigned int j=0 ; j<subSampleIndices.size() ; j++ )
				_addSample< KernelRadius >( iData , SampleDepth( subSampleIndices[j] ) + 1. , Sample( subSampleIndices[j] ).first , Sample( subSampleIndices[j] ).second , std::forward< DepthWeightFunctor >( DepthWeight ) );
	}
}

template< unsigned int Dim , typename DataType >
template< unsigned int KernelRadius , typename DepthWeightFunctor /* = std::function< Data ( unsigned int idx ) > */ >
void TreeSplatter< Dim , DataType >::_addSample( _InsertionData< KernelRadius > &iData , Point< double , Dim > p , DataType data , DepthWeightFunctor && DepthWeight )
{
	for( unsigned int d=1 ; d<=_maxDepth+1 ; d++ ) _addSample( iData , d , p , data * DepthWeight(d-1) );
}

template< unsigned int Dim , typename DataType >
template< unsigned int KernelRadius , typename DepthWeightFunctor /* = std::function< Data ( unsigned int idx ) > */ >
void TreeSplatter< Dim , DataType >::_addSample( _InsertionData< KernelRadius > &iData , double depth , Point< double , Dim > p , DataType data , DepthWeightFunctor && DepthWeight )
{
	if( depth>=(_maxDepth+1) ) _addSample( iData , _maxDepth+1 , p , data * DepthWeight( _maxDepth ) );
	else if( depth<=1 )        _addSample( iData ,           1 , p , data * DepthWeight(0) );
	else
	{
		unsigned int d1 = (unsigned int)floor(depth) , d2 = (unsigned int)floor(depth)+1;
		double dd = depth-d1;
		_addSample( iData , d1 , p , data * (1.-dd) * DepthWeight(d1-1) );
		_addSample( iData , d2 , p , data * (   dd) * DepthWeight(d2-1) );
	}
}

template< unsigned int Dim , typename DataType >
template< unsigned int KernelRadius >
void TreeSplatter< Dim , DataType >::_addSample( _InsertionData< KernelRadius > &iData , unsigned int depth , Point< double , Dim > p , DataType data )
{
	unsigned int res = 1<<(depth-1);
	Hat::Index< Dim > E;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		E[d] = (int)floor( p[d] * res );
		BSpline< _InsertionData< KernelRadius >::BSplineDegree >::Evaluate( p[d]*res-E[d] , iData.bSplineValues[d] );
	}

	typename NeighborKey< KernelRadius >::NeighborType &neighbors = iData.nKey.neighbors[depth];

	Hat::Index< Dim > offset;
	{
		offset = iData.nKey.center( depth )->offset();
		for( unsigned int d=0 ; d<Dim ; d++ ) offset[d] -= (int)KernelRadius;
	}

	double scl[Dim+1] ; bool inRange[Dim+1];
	scl[0] = 1. , inRange[0] = true;
	Window::Loop< Dim >::Run
	(
		0 , _InsertionData< KernelRadius >::BSplineDegree+1 ,
		[&]( int d , int i )
		{
			scl[d+1] = scl[d] * iData.bSplineValues[d][i];
			inRange[d+1] = inRange[d] && ( offset[d]+i>=0 && offset[d]+i<=(int)res );
		} , 
		[&]( Node *node ){ if( node && inRange[Dim] ) node->nodeData += scl[Dim] * data; } ,
		neighbors.neighbors
	);
}

template< unsigned int Dim , typename DataType >
template< unsigned int KernelRadius , typename PositionFunctor /* = std::function< Point< double , Dim > ( size_t idx ) > */  , typename DataFunctor /* = std::function< Data ( size_t idx ) > */ , typename DepthWeightFunctor /* = std::function< double ( size_t idx ) > */ >
void TreeSplatter< Dim , DataType >::_addSamples( NeighborKey< KernelRadius > &nKey , size_t sz , PositionFunctor && PositionF , DataFunctor && DataF , DepthWeightFunctor && DepthF )
{
	static const unsigned int BSplineDegree = 2*KernelRadius + 1;

	struct BSplineInfo
	{
		double values[Dim][ BSplineDegree+1 ];
		double scl[Dim+1] ;
	};

	std::vector< BSplineInfo > bSplineInfo( sz );

	for( unsigned int _depth=1 ; _depth<=_maxDepth+1 ; _depth++ )
	{
		double dWeight = DepthF( _depth-1 );

		unsigned int res = 1<<(_depth-1);
		for( size_t j=0 ; j<sz ; j++ )
		{
			Hat::Index< Dim > E;
			Point< double , Dim > p = PositionF( j );
			for( unsigned int d=0 ; d<Dim ; d++ )
			{
				E[d] = (int)floor( p[d] * res );
				BSpline< _InsertionData< KernelRadius >::BSplineDegree >::Evaluate( p[d]*res-E[d] , bSplineInfo[j].values[d] );
			}
			bSplineInfo[j].scl[0] = dWeight;
		}

		typename NeighborKey< KernelRadius >::NeighborType &neighbors = nKey.neighbors[_depth];

		Hat::Index< Dim > offset;
		{
			offset = nKey.center( _depth )->offset();
			for( unsigned int d=0 ; d<Dim ; d++ ) offset[d] -= (int)KernelRadius;
		}

		bool inRange[Dim+1];
		inRange[0] = true;
		Window::Loop< Dim >::Run
		(
			0 , BSplineDegree+1 ,
			[&]( int d , int i )
			{
				for( size_t j=0 ; j<sz ; j++ ) bSplineInfo[j].scl[d+1] = bSplineInfo[j].scl[d] * bSplineInfo[j].values[d][i];
				inRange[d+1] = inRange[d] && ( offset[d]+i>=0 && offset[d]+i<=(int)res );
			} , 
			[&]( Node *node )
			{ 
				if( node && inRange[Dim] ) for( size_t j=0 ; j<sz ; j++ ) node->nodeData += DataF( j ) * bSplineInfo[j].scl[Dim];
			} ,
			neighbors.neighbors
		);
	}
}

template< unsigned int Dim , typename DataType >
DataType TreeSplatter< Dim , DataType >::operator()( Point< double , Dim > p , unsigned int depth , unsigned int thread ) const
{
	static const unsigned int KernelRadius = 0;
	static const unsigned int BSplineDegree = 2*KernelRadius+1;

	ConstNeighborKey< KernelRadius > &nKey = const_cast< ConstNeighborKey< KernelRadius >& >( _nKeys[thread] );
	{
		const Node *n = _spaceRoot->getLeafNode( p );
		if( !n || n->depth()<(depth+1) ) MK_ERROR_OUT( "Couldn't find node: " , p , " @ " , depth );
		nKey.getNeighbors( n );
	}
	typename ConstNeighborKey< KernelRadius >::NeighborType &neighbors = nKey.neighbors[depth+1];

	double bSplineValues[ Dim ][ BSplineDegree+1 ];

	DataType data{};

	unsigned int res = 1<<depth;
	Hat::Index< Dim > e;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		e[d] = (int)floor( p[d] * res );
		BSpline< BSplineDegree >::Evaluate( p[d]*res - e[d] , bSplineValues[d] );
	}

	Hat::Index< Dim > offset;
	{
		offset = nKey.center( depth )->offset();
		for( unsigned int d=0 ; d<Dim ; d++ ) offset[d] -= (int)KernelRadius;
	}

	double scl[Dim+1] ; bool inRange[Dim+1];
	scl[0] = 1. , inRange[0] = true;
	Window::Loop< Dim >::Run
	(
		0 , _InsertionData< KernelRadius >::BSplineDegree+1 ,
		[&]( int d , int i )
		{
			scl[d+1] = scl[d] * bSplineValues[d][i];
			inRange[d+1] = inRange[d] && ( offset[d]+i>=0 && offset[d]+i<=(int)res );
		} , 
		[&]( const Node *node ){ if( node && inRange[Dim] ) data += node->nodeData * scl[Dim]; } ,
		neighbors.neighbors
	);

	return data;
}

template< unsigned int Dim , typename DataType >
DataType TreeSplatter< Dim , DataType >::operator()( Point< double , Dim > p , unsigned int thread ) const
{
	static const unsigned int KernelRadius = 0;
	static const unsigned int BSplineDegree = 2*KernelRadius+1;

	unsigned int _maxDepth;
	ConstNeighborKey< KernelRadius > &nKey = const_cast< ConstNeighborKey< KernelRadius >& >( _nKeys[thread] );
	{
		const Node *n = _spaceRoot->getLeafNode( p );
		_maxDepth = n->depth();
		nKey.getNeighbors( n );
	}

	double bSplineValues[ Dim ][ BSplineDegree+1 ];

	DataType data{};

	for( unsigned int _depth=1 ; _depth<=_maxDepth ; _depth++ )
	{
		typename ConstNeighborKey< KernelRadius >::NeighborType &neighbors = nKey.neighbors[_depth];
		unsigned int res = 1<<(_depth-1);
		Hat::Index< Dim > e;
		for( unsigned int d=0 ; d<Dim ; d++ )
		{
			e[d] = (int)floor( p[d] * res );
			BSpline< BSplineDegree >::Evaluate( p[d]*res - e[d] , bSplineValues[d] );
		}

		Hat::Index< Dim > offset;
		{
			offset = nKey.center( _depth )->offset();
			for( unsigned int d=0 ; d<Dim ; d++ ) offset[d] -= (int)KernelRadius;
		}

		double scl[Dim+1] ; bool inRange[Dim+1];
		scl[0] = 1. , inRange[0] = true;
		Window::Loop< Dim >::Run
		(
			0 , _InsertionData< KernelRadius >::BSplineDegree+1 ,
			[&]( int d , int i )
			{
				scl[d+1] = scl[d] * bSplineValues[d][i];
				inRange[d+1] = inRange[d] && ( offset[d]+i>=0 && offset[d]+i<=(int)res );
			} , 
			[&]( const Node *node ){ if( node && inRange[Dim] ) data += node->nodeData * scl[Dim]; } ,
			neighbors.neighbors
		);
	}

	return data;
}


// Processes the data in the trees
template< unsigned int Dim , typename DataType >
template< typename DataFunctor /* = std::function< void ( unsigned int , Hat::Index< Dim > , const DataType & ) > */ >
void TreeSplatter< Dim , DataType >::process( DataFunctor && F ) const
{
	static_assert( std::is_convertible_v< DataFunctor , std::function< void ( unsigned int , Hat::Index< Dim > , const DataType & ) > > , "[ERROR] DataFunctor poorly formed" );

	auto _F = [&]( const Node *node )
		{
			int depth;
			Hat::Index< Dim > offset;
			node->depthAndOffset( depth , offset );
			if( depth>0 )
			{
				depth--;
				unsigned int res = 1<<depth;
				bool interior = true;
				for( unsigned int d=0 ; d<Dim ; d++ ) interior &= ( 0<=offset[d] && offset[d]<=(int)res );
				if( interior ) F( (unsigned int)depth , offset , node->nodeData );
			}
		};
	_root.processNodes( _F );
}

template< unsigned int Dim , typename DataType >
template< typename DataFunctor /* = std::function< void ( unsigned int , Hat::Index< Dim > , DataType & ) > */ >
void TreeSplatter< Dim , DataType >::process( DataFunctor && F )
{
	static_assert( std::is_convertible_v< DataFunctor , std::function< void ( unsigned int , Hat::Index< Dim > , DataType & ) > > , "[ERROR] DataFunctor poorly formed" );

	auto _F = [&]( Node *node )
		{
			int depth;
			Hat::Index< Dim > offset;
			node->depthAndOffset( depth , offset );
			if( depth>0 )
			{
				depth--;
				unsigned int res = 1<<depth;
				bool interior = true;
				for( unsigned int d=0 ; d<Dim ; d++ ) interior &= ( 0<=offset[d] && offset[d]<=(int)res );
				if( interior ) F( (unsigned int)depth , offset , node->nodeData );
			}
		};
	_root.processNodes( _F );
}

template< unsigned int Dim , typename DataType >
void TreeSplatter< Dim , DataType >::write( BinaryStream &stream , bool serialize ) const { _root.write( stream , serialize ); }

template< unsigned int Dim , typename DataType >
void TreeSplatter< Dim , DataType >::read( BinaryStream &stream )
{
	_root.read( stream , nullptr );
	_maxDepth = _root.maxDepth()-1;
	_spaceRoot = _root.children;
	_nKeys.resize( ThreadPool::NumThreads() );
	for( unsigned int i=0 ; i<_nKeys.size() ; i++ ) _nKeys[i].set( _maxDepth+1 );
}

///////////////////
// TreeEstimator //
///////////////////
template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
TreeEstimator< Dim , CoDim , HasNoise >::TreeEstimator( void ) : TreeSplatter< Dim , Info >( 0 ){}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
TreeEstimator< Dim , CoDim , HasNoise >::TreeEstimator( BinaryStream &stream ) : TreeSplatter< Dim , Info >( stream )
{
	stream.read( _targetSamplesPerCell );
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
void TreeEstimator< Dim , CoDim , HasNoise >::read( BinaryStream &stream )
{
	TreeSplatter< Dim , Info >::read( stream );
	stream.read( _targetSamplesPerCell );
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
void TreeEstimator< Dim , CoDim , HasNoise >::write( BinaryStream &stream , bool serialize ) const
{
	TreeSplatter< Dim , Info >::write( stream , serialize );
	stream.write( _targetSamplesPerCell );
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
template< typename SampleFunctor /* = std::function< Point< double , Dim > > ( unsigned int idx ) */ >
TreeEstimator< Dim , CoDim , HasNoise >::TreeEstimator( unsigned int kernelRadius , unsigned int maxDepth , SampleFunctor && F , const OrderedSampler< Dim > &orderedSampler , bool mergeSamples , double targetSamplesPerCell )
	: TreeSplatter< Dim , Info >( maxDepth )
{
	static_assert( std::is_convertible_v< SampleFunctor , std::function< Point< double , Dim > ( unsigned int ) > > , "[ERROR] SampleFunctor poorly formed" );

	_targetSamplesPerCell = targetSamplesPerCell;

	std::vector< double > scales( _maxDepth+1 );
	{
		double scale;
		{
			unsigned int bSplineDegree = 2*kernelRadius + 1;
			std::vector< double > _values( bSplineDegree+1 );
			BSpline<>::Evaluate( bSplineDegree , 0.5 , &_values[0] );
			scale = pow( 1./_values[ kernelRadius + 1 ] , CoDim );
		}
		for( unsigned int d=0 ; d<=_maxDepth ; d++ ) scales[d] = scale / pow( pow( 0.5 , d ) , Dim-CoDim );
	}
	auto DepthF = [&scales]( unsigned int d ){ return scales[d]; };
	auto _F = [&]( size_t idx ){ return std::pair< Point< double , Dim > , Info >( F(idx) , Info( F(idx) , 1. ) ); };
	switch( kernelRadius )
	{
	case 0: TreeSplatter< Dim , Info >::template addSamples< 0 >( _F , orderedSampler , DepthF , mergeSamples ) ; break;
	case 1: TreeSplatter< Dim , Info >::template addSamples< 1 >( _F , orderedSampler , DepthF , mergeSamples ) ; break;
	case 2: TreeSplatter< Dim , Info >::template addSamples< 2 >( _F , orderedSampler , DepthF , mergeSamples ) ; break;
	case 3: TreeSplatter< Dim , Info >::template addSamples< 3 >( _F , orderedSampler , DepthF , mergeSamples ) ; break;
	case 4: TreeSplatter< Dim , Info >::template addSamples< 4 >( _F , orderedSampler , DepthF , mergeSamples ) ; break;
	default: MK_ERROR_OUT( "Only radii [0,4] supported: " , kernelRadius );
	}
}


template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
typename TreeEstimator< Dim , CoDim , HasNoise >::Info TreeEstimator< Dim , CoDim , HasNoise >::_info( ConstNeighborKey< 0 > &nKey , unsigned int depth , Point< double , Dim > p ) const
{
	static const unsigned int KernelRadius = 0;
	static const unsigned int BSplineDegree = 2*KernelRadius+1;
	double values[ Dim ][ BSplineDegree+1 ];

	Info info;

	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.first[d] = 0 , range.second[d] = BSplineDegree + 1;

	unsigned int res = 1<<(depth-1);
	Hat::Index< Dim > e;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		e[d] = (int)floor( p[d] * res );
		BSpline< BSplineDegree >::Evaluate( p[d]*res - e[d] , values[d] );
	}

	typename ConstNeighborKey< KernelRadius >::NeighborType &neighbors = nKey.neighbors[depth];
	auto f = [&]( Hat::Index< Dim > _f )
		{
			const Node *nbr = neighbors.neighbors( &_f[0] );
			if( nbr )
			{
				Hat::Index< Dim > f = _f + e;
				double scale = 1.;
				for( unsigned int d=0 ; d<Dim ; d++ ) scale *= values[d][ _f[d] ];
				info += nbr->nodeData * scale;
			}
		};
	range.process(f);

	return info;
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
double TreeEstimator< Dim , CoDim , HasNoise >::_samplesPerCell( ConstNeighborKey< 0 > &nKey , unsigned int depth , Point< double , Dim > p ) const
{
	return _info( nKey , depth , p ).weightSum / ( 1<<((depth-1)*(Dim-CoDim) ) );
}


template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
typename TreeEstimator< Dim , CoDim , HasNoise >::Info TreeEstimator< Dim , CoDim , HasNoise >::operator()( Point< double , Dim > p , unsigned int thread ) const
{
	ConstNeighborKey< 0 > &nKey = const_cast< ConstNeighborKey< 0 >& >( _nKeys[thread] );
	{
		const Node *n = _spaceRoot->getLeafNode( p );
		if( !n || n->depth()!=nKey.depth() ) MK_ERROR_OUT( "Couldn't find node: " , p );
		nKey.getNeighbors( n );
	}
	double depth = _depth( nKey , nKey.depth() , p );
	if( depth>=nKey.depth() ) return _info( nKey , nKey.depth() , p );
	else if( depth<0 ) return _info( nKey , 1 , p );
	else
	{
		int d1 = (int)floor(depth) , d2 = (int)floor(depth)+1;
		double d = depth-d1;
		return _info( nKey , d1 , p ) * (1.-d) + _info( nKey , d2 , p ) * d;
	}
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
double TreeEstimator< Dim , CoDim , HasNoise >::measure( Point< double , Dim > p , unsigned int thread ) const { return operator()( p , thread ).measure(); }

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
double TreeEstimator< Dim , CoDim , HasNoise >::noise( Point< double , Dim > p , unsigned int thread ) const
{
	if constexpr( HasNoise ) return operator()( p , thread ).noise( CoDim );
	else                     return 1.;
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
double TreeEstimator< Dim , CoDim , HasNoise >::samplesPerCell( Point< double , Dim > p , unsigned int thread ) const
{
	// If at depth d, the cell has _targetSamplesPerCell samples per cell then,
	// assuming that the number samples per cell decreases by a factor of 2^{Dim-CoDim} per level,
	// the number of samples per cell at depth _maxDepth is _targetSamplesPerCell * pow( 0.5 , (Dim-CoDim) * (_maxDepth-d)
	return _targetSamplesPerCell * pow( 0.5 , (Dim-CoDim) * ( (int)_maxDepth - depth( p , thread ) ) );
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
double TreeEstimator< Dim , CoDim , HasNoise >::depth( Point< double , Dim > p , unsigned int thread ) const
{
	ConstNeighborKey< 0 > &nKey = const_cast< ConstNeighborKey< 0 >& >( _nKeys[thread] );
	const Node *n = _spaceRoot->getLeafNode( p );
	if( !n ) MK_ERROR_OUT( "Couldn't find node: " , p );
//		if( !n || n->depth()!=nKey.depth() ) MK_ERROR_OUT( "Couldn't find node: " , p );
	nKey.getNeighbors( n );
	return _depth( nKey , n->depth() , p ) - 1.;
}

template< unsigned int Dim , unsigned int CoDim , bool HasNoise >
double TreeEstimator< Dim , CoDim , HasNoise >::_depth( ConstNeighborKey< 0 > &nKey , unsigned int _maxDepth , Point< double , Dim > p ) const
{
	// Find the finest depth at which the number of samples per node exceeds _targetSamplesPerCell
	unsigned int d1 = _maxDepth , d2 = _maxDepth+1;
	while( d2>1 && _samplesPerCell( nKey , d1 , p )<_targetSamplesPerCell ) d1-- , d2--;
	if( d2==1 || d2==_maxDepth+1 ) // Extremal case
	{
		unsigned int d;
		if     ( d2==1 ) d=1;
		else if( d2==_maxDepth+1 ) d = _maxDepth;
		double samplesPerCell = _samplesPerCell( nKey , d , p );
		// Generically the assumption is that increase the depth by should decrease the number of samples per cell by a factor of 2^{Dim-CoDim}
		// Solve for the depth such that:
		//		_samplesPerCell = samplesPerCell / 2^{depth-d}
		//		2^{depth-d} = samplesPerCell/_samplesPerCell
		//		depth - d = log( samplesPerCell/_samplesPerCell ) / log(2)
		//		depth = log( samplesPerCell/_samplesPerCell ) / log(2) + d
		return d + log( samplesPerCell/_targetSamplesPerCell ) / log(2.);
	}
	else // Interior case
	{
		// Here we assume that the change in the samples per cell is simply samplesPerCell[d2]/samplesPerCell[d1]
		// Solve for the depth such that:
		//		_samplesPerCell = samplesPerCell[d1] * pow( samplesPerCell[d2]/samplesPerCell[d1] , d-d1 )
		//		pow( samplesPerCell[d2]/samplesPerCell[d1] , d-d1 ) = _samplesPerCell / samplesPerCell[d1]
		//		d-d1 = log( _samplesPerCell / samplesPerCell[d1] ) / log( samplesPerCell[d2]/samplesPerCell[d1] )
		//		d = log( _samplesPerCell / samplesPerCell[d1] ) / log( samplesPerCell[d2]/samplesPerCell[d1] ) + d1
		double samplesPerCell1 = _samplesPerCell( nKey , d1 , p) , samplesPerCell2 = _samplesPerCell( nKey , d2 , p );
		if( !samplesPerCell2 ) return d1;
		else return d1 + log( _targetSamplesPerCell/samplesPerCell1 ) / log( samplesPerCell2/samplesPerCell1 );
	}
}
