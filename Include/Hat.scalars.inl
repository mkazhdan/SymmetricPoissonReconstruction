/////////////////////////////////////////
// ScalarFunctions::IntegrationStencil //
/////////////////////////////////////////
template< unsigned int Dim >
template< typename T , unsigned int Rank , unsigned int Radius >
template< typename ... F >
T ScalarFunctions< Dim >::IntegrationStencil< T , Rank , Radius >::operator()( Index< Dim > e , F ... f ) const
{
	static_assert( sizeof...(f)==Rank , "[ERROR] Index count does not match rank" );
	Index< Rank > idx( Point< int , Rank >( Basis< Dim >::template _RelativeIndex< Radius >( e , f )... ) );
	return SquareStencil< T , Rank , StencilSize< Radius >() >::operator()( idx );
}

template< unsigned int Dim >
template< typename T , unsigned int Rank , unsigned int Radius >
template< typename ... F >
T ScalarFunctions< Dim >::IntegrationStencil< T , Rank , Radius >::operator()( Range< Dim > eRange , F ... f ) const
{
	static_assert( sizeof...(f)==Rank , "[ERROR] Index count does not match rank" );
	T value{};
	Range< Dim >::Intersect( Basis< Dim >::FunctionSupport( f ).dilate( Radius ) ... , eRange ).process( [&]( Index< Dim > e ){ value += this->operator()( e , f... ); } );
	return value;
}

///////////////////////////////////////
// ScalarFunctions::StencilCaseTable //
///////////////////////////////////////
template< unsigned int Dim >
template< unsigned int Radius >
Hat::Range< Dim > ScalarFunctions< Dim >::StencilCaseTable< Radius >::Range( unsigned int res )
{
	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = std::min< unsigned int >( Width , res+1 );
	return range;
}

template< unsigned int Dim >
template< unsigned int Radius >
size_t ScalarFunctions< Dim >::StencilCaseTable< Radius >::Size( Hat::Index< Dim > C , unsigned int res )
{
	size_t sz = 1;
	if( res>Width ) for( unsigned int d=0 ; d<Dim ; d++ ) if( C[d]==Radius ) sz *= ( res+1 - 2*Radius );
	return sz;
}

template< unsigned int Dim >
template< unsigned int Radius >
size_t ScalarFunctions< Dim >::StencilCaseTable< Radius >::SubIndex( Hat::Index< Dim > I , unsigned int res )
{
	size_t idx = 0;
	if( res>Width )
	{
		Index< Dim > C = IndexToCase( I , res );
		unsigned int _subWidth = 1;
		for( unsigned int d=0 ; d<Dim ; d++ ) if( C[d]==Radius )
		{
			idx += ( I[d]-Radius ) * _subWidth;
			_subWidth *= res+1 - 2*Radius;
		}
	}
	return idx;
}

template< unsigned int Dim >
template< unsigned int Radius >
Index< Dim > ScalarFunctions< Dim >::StencilCaseTable< Radius >::CaseToIndex( Index< Dim > C , unsigned int res )
{
	if( res>Width )
	{
		for( unsigned int d=0 ; d<Dim ; d++ )
		{
			if     ( C[d]<(int)Radius );
			else if( C[d]>(int)Radius ) C[d] = C[d] - Radius + (res-Radius);
			else                        C[d] = res/2;
		}
	}
	return C;
}

template< unsigned int Dim >
template< unsigned int Radius >
Index< Dim > ScalarFunctions< Dim >::StencilCaseTable< Radius >::IndexToCase( Index< Dim > I , unsigned int res )
{
	if( res>Width )
	{
		for( unsigned int d=0 ; d<Dim ; d++ )
		{
			if     ( I[d]<(int)(    Radius) );
			else if( I[d]>(int)(res-Radius) ) I[d] = Radius + I[d] - (res-Radius);
			else                              I[d] = Radius;
		}
	}
	return I;
}

/////////////////////////////////
// ScalarFunctions::MatrixInfo //
/////////////////////////////////
template< unsigned int Dim >
template< unsigned int Radius , bool Sym >
ScalarFunctions< Dim >::MatrixInfo< Radius , Sym >::MatrixInfo( unsigned int res ) : _caseRange( StencilCaseTable< Radius >::Range( res ) ) , _res(res)
{
	for( unsigned int d=0 ; d<Dim ; d++ ) _fRange.second[d] = _res+1 , _off[d] = Radius , _infoRange.second[d] = 2*Radius+1;
	size_t offset = 0;

	auto f = [&]( Index< Dim > C )
	{
		// Get an index corresponding to the case
		Index< Dim > f1 = StencilCaseTable< Radius >::CaseToIndex( C , res );

		// The information for this case
		_Info &info = _info(C);
		info.sizes[0] = info.sizes[1] = 0;
		info.offset = offset;

		// First clear
		_infoRange.process( [&]( Index< Dim > I ){ info.stencil(I) = -1; } );

		// Then set
		{
			auto f = [&]( Index< Dim > f2 )
			{
				if( f1<f2 || ( f1==f2 && Sym ) ) info.stencil( f2 - f1 + _off ) = info.sizes[0]++;
				if( f1>f2 ) info.sizes[1]++;
			};
			Range< Dim >::Intersect( Range< Dim >( f1 ).dilate( Radius ) , _fRange ).process( f );
			offset += StencilCaseTable< Radius >::Size( C , _res ) * info.sizes[0];
		}
	};
	// Iterate over the different cases and set the associated information
	_caseRange.process( f );
}

template< unsigned int Dim >
template< unsigned int Radius , bool Sym >
size_t ScalarFunctions< Dim >::MatrixInfo< Radius , Sym >::entries( bool all ) const
{
	size_t e = 0;

	auto f = [&]( Index< Dim > C )
	{
		e += _info(C).sizes[0] * StencilCaseTable< Radius >::Size( C , _res );
		if( all ) e += _info(C).sizes[1] * StencilCaseTable< Radius >::Size( C , _res );
	};
	_caseRange.process( f );

	return e;
}

template< unsigned int Dim >
template< unsigned int Radius , bool Sym >
size_t ScalarFunctions< Dim >::MatrixInfo< Radius , Sym >::entry( Index< Dim > F1 , Index< Dim > F2 ) const
{
	const _Info &info = _info( StencilCaseTable< Radius >::IndexToCase( F1 , _res ) );
	Index< Dim > I = F2 - F1 + _off;
	if( !_infoRange.contains(I) ) MK_ERROR_OUT( "Bad index pair: " , F1 , " : " , F2 );
	size_t e = info.stencil(I);
	if( e==-1 ) MK_ERROR_OUT( "No entry: " , F1 , " : " , F2 );
	return info.offset + StencilCaseTable< Radius >::SubIndex( F1 , _res ) * info.sizes[0] + e;
}

template< unsigned int Dim >
template< unsigned int Radius , bool Sym >
template< typename F /* = std::function< void ( Index< Dim > , size_t ) > */ >
void ScalarFunctions< Dim >::MatrixInfo< Radius , Sym >::process( Index< Dim > f1 , F f ) const
{
	const _Info &info = _info( StencilCaseTable< Radius >::IndexToCase( f1 , _res ) );
	auto _f = [&]( Index< Dim > I )
	{
		size_t e = info.stencil(I);
		if( e!=-1 )
		{
			Index< Dim > f2 = f1 + I - _off;
			if( !_fRange.contains( f2 ) ) MK_ERROR_OUT( "should not be happening" );
			f( f2 , info.offset + StencilCaseTable< Radius >::SubIndex( f1 , _res ) * info.sizes[0] + e );
		}
	};
	_infoRange.process( _f );
}

template< unsigned int Dim >
template< unsigned int Radius , bool Sym >
template< typename F /* = std::function< void ( Index< Dim > , size_t , bool ) > */ >
void ScalarFunctions< Dim >::MatrixInfo< Radius , Sym >::processAll( Index< Dim > f1 , F f ) const
{
	const _Info &info = _info( StencilCaseTable< Radius >::IndexToCase( f1 , _res ) );

	auto _f = [&]( Index< Dim > I )
	{
		size_t e = info.stencil(I);
		Index< Dim > f2 = f1 + I - _off;
		// Confirm that this is an in-bounds function index
		if( !_fRange.contains( f2 ) ) return;
		if( e!=-1 ) f( f2 , info.offset + StencilCaseTable< Radius >::SubIndex( f1 , _res ) * info.sizes[0] + e , true );
		else
		{
			// I = (f2-f1) + _off
			// Flipping the roles of f1 and f2
			// I <- (f1-f2) + _off = - ( I - _off ) + _off
			const _Info &info = _info( StencilCaseTable< Radius >::IndexToCase( f2 , _res ) );
			// The index of f1 relative to f2
			I = - ( I - _off ) + _off;
			e = info.stencil(I);
			if( e!=-1 ) f( f2 , info.offset + StencilCaseTable< Radius >::SubIndex( f2 , _res ) * info.sizes[0] + e , false );
		}
	};
	// Process _all_ functions within the prescribed radius
	_infoRange.process( _f );
}

template< unsigned int Dim >
template< unsigned int Radius , bool Sym >
size_t ScalarFunctions< Dim >::MatrixInfo< Radius , Sym >::entries( Index< Dim > I , bool all ) const
{
	if( all ) return _info( StencilCaseTable< Radius >::IndexToCase( I , _res ) ).sizes[0] + _info( StencilCaseTable< Radius >::IndexToCase( I , _res ) ).sizes[1];
	else      return _info( StencilCaseTable< Radius >::IndexToCase( I , _res ) ).sizes[0];
}

/////////////////////////////////////////////
// ScalarFunctions::FullIntegrationStencil //
/////////////////////////////////////////////
template< unsigned int Dim >
template< typename T , unsigned int Radius >
constexpr unsigned int ScalarFunctions< Dim >::FullIntegrationStencil< T , Radius >::StencilNum( void )
{
	constexpr unsigned int Size = 3 + 4*Radius;
	if constexpr( Dim==1 ) return Size;
	else return ScalarFunctions< Dim-1 >::template FullIntegrationStencil< T , Radius >::StencilNum() * Size;
}

template< unsigned int Dim >
template< typename T , unsigned int Radius>
ScalarFunctions< Dim >::FullIntegrationStencil< T , Radius >::FullIntegrationStencil( void ) : _res(-1){}

template< unsigned int Dim >
template< typename T , unsigned int Radius>
ScalarFunctions< Dim >::FullIntegrationStencil< T , Radius >::FullIntegrationStencil( const IntegrationStencil< T , 2 , Radius > &stencil , unsigned int res ) : _res( res ) , _rows( StencilNum() )
{
	Hat::Index< Dim > Off;
	Range< Dim > eRange , fRange , range;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1 , eRange.first[d] = 0 , eRange.second[d] = _res , fRange.first[d] = 0 , fRange.second[d] = _res+1 , range.first[d] = 0 , range.second[d] = 3+4*Radius;

	auto f = [&]( Index< Dim > f1 )
	{
		// [0,2*Radius]         <-> [0,2*Radius]
		// [_res/2]             <-> [2*Radius+1]
		// [_res-2*Radius,_res] <-> [2*Radius+2,4*Radius+2]
		for( unsigned int d=0 ; d<Dim ; d++ )
			if     ( f1[d]<=2*Radius   );
			else if( f1[d]==2*Radius+1 ) f1[d] = _res/2;
			else                         f1[d] = _res - 4*Radius-2 + f1[d];
		Row _row;
		std::map< Index< Dim > , T > row;

		auto f = [&]( Index< Dim > e )
		{
			auto f = [&]( Index< Dim > f2 ){ row[ f2-f1 ] += stencil( e , f1 , f2 ); };
			Range< Dim >::Intersect( Basis< Dim >::ElementSupport( e ).dilate( Radius ) , fRange ).process(f);
//			Basis< Dim >::ElementSupport( e ).dilate( Radius ).process(f);
		};
		Range< Dim >::Intersect( Basis< Dim >::FunctionSupport( f1 ).dilate( Radius ) , eRange ).process( f );

		_row.reserve( row.size() );
		for( auto iter=row.begin() ; iter!=row.end() ; iter++ ) _row.push_back( Entry( iter->first , ScalarFunctions< Dim >::FunctionIndex( iter->first + Off , 2 ) , iter->second ) );

		_rows[ StencilIndex(f1,_res) ] = _row;
	};
	Range< Dim >::Intersect( range , fRange ).process( f );
}

template< unsigned int Dim >
template< typename T , unsigned int Radius >
unsigned int ScalarFunctions< Dim >::FullIntegrationStencil< T , Radius >::StencilIndex( Index< Dim > f , unsigned int res )
{
	unsigned int idx=0;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		if( f[d]<0 || f[d]>(int)res ) MK_ERROR_OUT( "Bad function index: " , f , " / " , res );
		if     ( f[d]<=(int)(    2*Radius) ) idx = idx*(3+4*Radius) + f[d];
		else if( f[d]>=(int)(res-2*Radius) ) idx = idx*(3+4*Radius) + 2*Radius+2+f[d]-(res-2*Radius);
		else                                 idx = idx*(3+4*Radius) + 2*Radius+1;
	}
	return idx;
}

template< unsigned int Dim >
template< typename T , unsigned int Radius >
const typename ScalarFunctions< Dim >::template FullIntegrationStencil< T , Radius >::Row &ScalarFunctions< Dim >::FullIntegrationStencil< T , Radius >::row( Index< Dim > f ) const
{
	return _rows[ StencilIndex( f , _res ) ];
}

template< unsigned int Dim >
template< typename T >
typename ScalarFunctions< Dim >::template FullIntegrationStencil< T , 0 > ScalarFunctions< Dim >::Restrict( const FullIntegrationStencil< T , 0 > &stencilF )
{
	if( stencilF._res&1 ) MK_ERROR_OUT( "Cannot restrict when reslution is odd" );
	FullIntegrationStencil< T , 0 > stencilC;
	stencilC._res = stencilF._res/2;
	stencilC._rows.resize( FullIntegrationStencil< T , 0 >::StencilNum() );


	Range< Dim > fRange , range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 3 , fRange.second[d] = stencilC._res+1;

	auto f = [&]( Index< Dim > F_coarse1 )
		{
			// Transform the case-index to a coarse index
			// [0]      <-> [0]
			// [_res/2] <-> [1]
			// [_res]   <-> [2]
			for( unsigned int d=0 ; d<Dim ; d++ )
				if     ( F_coarse1[d]==0 );
				else if( F_coarse1[d]==1 ) F_coarse1[d] = stencilC._res/2;
				else                       F_coarse1[d] = stencilC._res;

			std::map< Index< Dim > , T > row;

			auto f = [&]( Index< Dim > F_fine1 , double weight1 )
				{
					// Iterate over all stencil entries supported on F_fine1
					const typename FullIntegrationStencil< T , 0 >::Row &rowF = stencilF.row( F_fine1 );
					for( unsigned int i=0 ; i<rowF.size() ; i++ )
					{
						Index< Dim > F_fine2 = F_fine1 + rowF[i].first;
						T value = rowF[i].second * weight1;

						auto f = [&]( Index< Dim > F_coarse2 , double weight2 ){ row[ F_coarse2-F_coarse1 ] += value * weight2; };
						// Iterate over all the coarser nodes F_fine2 restricts to
						Restriction::Process( F_fine2 , f );
					}
				};
			// Iterate over all the finer nodes F_coarse1 prolongs to
			Prolongation::Process( F_coarse1 , stencilC._res , f );

			typename FullIntegrationStencil< T , 0 >::Row &_row = stencilC._rows[ FullIntegrationStencil< T , 0 >::StencilIndex( F_coarse1 , stencilC._res ) ];

			_row.reserve( row.size() );
			for( auto iter=row.begin() ; iter!=row.end() ; iter++ ) _row.push_back( typename FullIntegrationStencil< T , 0 >::Entry( iter->first , iter->second ) );
		};
	// Iterate over the cases
	Range< Dim >::Intersect( range , fRange ).process( f );
	return stencilC;
}

//////////////////////////////////////////
// ScalarFunctions::ProlongationStencil //
//////////////////////////////////////////
template< unsigned int Dim >
ScalarFunctions< Dim >::ProlongationStencil::ProlongationStencil( void )
{
	Index< Dim > off;
	for( unsigned int d=0 ; d<Dim ; d++ ) _range.first[d] = -1 , _range.second[d] = 2 , off[d] = 1;

	auto f = [&]( Index< Dim > I )
	{
		double value = 1;
		for( unsigned int d=0 ; d<Dim; d++ ) if( I[d]!=0 ) value *= 0.5;
		SquareStencil< double , Dim , StencilCaseTable<1>::Width >::operator()( I+off ) = value;
	};
	_range.process( f );
}

//////////////////
// Prolongation //
//////////////////
template< unsigned int Dim >
template< typename ProlongationFunctor/*=std::function< void ( Index< Dim > F_fine , double weight ) >*/ >
void ScalarFunctions< Dim >::Prolongation::Process( Index< Dim > F_coarse , unsigned int coarseRes , ProlongationFunctor pFunctor )
{
	Range< Dim > fRange;
	for( unsigned int d=0 ; d<Dim ; d++ ) fRange.second[d] = 2*coarseRes+1;
	auto f = [&]( Index< Dim > F_fine )
		{
			double scale = 1.;
			for( unsigned int d=0 ; d<Dim ; d++ ) if( F_fine[d]!=(F_coarse[d]*2) ) scale *= 0.5;
			pFunctor( F_fine , scale );
		};
	Range< Dim >::Intersect( Range< Dim >( F_coarse*2 ).dilate(1) , fRange ).process( f );
}

/////////////////
// Restriction //
/////////////////
template< unsigned int Dim >
template< typename RestrictionFunctor/*=std::function< void ( Index< Dim > F_coarse , double weight ) >*/ >
void ScalarFunctions< Dim >::Restriction::Process( Index< Dim > F_fine , RestrictionFunctor rFunctor )
{
	Index< Dim > F_coarse;
	_Process< 0 >( F_fine , F_coarse , 1. , rFunctor );
}

template< unsigned int Dim >
template< unsigned int D , typename RestrictionFunctor/*=std::function< void ( Index< Dim > F_coarse , double weight ) >*/ >
void ScalarFunctions< Dim >::Restriction::_Process( Index< Dim > F_fine , Index< Dim > F_coarse , double weight , RestrictionFunctor &rFunctor )
{
	if constexpr( D==Dim ) rFunctor( F_coarse , weight );
	else
	{
		F_coarse[D] = F_fine[D]>>1;
		if( F_fine[D]&1 )
		{
			weight *= 0.5;
			_Process< D+1 >( F_fine , F_coarse , weight , rFunctor );
			F_coarse[D]++;
			_Process< D+1 >( F_fine , F_coarse , weight , rFunctor );
		}
		else _Process< D+1 >( F_fine , F_coarse , weight , rFunctor );
	}
}

/////////////////////
// ScalarFunctions //
/////////////////////
template< unsigned int Dim >
size_t ScalarFunctions< Dim >::FunctionNum( unsigned int res )
{
	size_t sz = 1;
	for( unsigned int d=0 ; d<Dim ; d++ ) sz *= res+1;
	return sz;
}

template< unsigned int Dim >
size_t ScalarFunctions< Dim >::ElementNum( unsigned int res )
{
	size_t sz = 1;
	for( unsigned int d=0 ; d<Dim ; d++ ) sz *= res;
	return sz;
}

template< unsigned int Dim >
size_t ScalarFunctions< Dim >::ElementIndex( Index< Dim > i , unsigned int res )
{
	size_t idx=0;
	for( unsigned int d=0 ; d<Dim ; d++ ) idx = idx*res + i[Dim-d-1];
	return idx;
}

template< unsigned int Dim >
Index< Dim > ScalarFunctions< Dim >::ElementIndex( size_t i , unsigned int res )
{
	Index< Dim > idx;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		idx[d] = i % res;
		i /= res;
	}
	return idx;
}

template< unsigned int Dim >
size_t ScalarFunctions< Dim >::FunctionIndex( Index< Dim > i , unsigned int res )
{
	size_t idx=0;
	for( unsigned int d=0 ; d<Dim ; d++ ) idx = idx*(res+1) + i[Dim-d-1];
	return idx;
}

template< unsigned int Dim >
Index< Dim > ScalarFunctions< Dim >::FunctionIndex( size_t i , unsigned int res )
{
	Index< Dim > idx;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		idx[d] = i % (res+1);
		i /= res+1;
	}
	return idx;
}

template< unsigned int Dim >
Range< Dim > ScalarFunctions< Dim >::ElementRange( unsigned int res )
{
	Range< Dim > eRange;
	for( unsigned int d=0 ; d<Dim ; d++ ) eRange.first[d] = 0 , eRange.second[d] = res;
	return eRange;
}

template< unsigned int Dim >
Range< Dim > ScalarFunctions< Dim >::FunctionRange( unsigned int res )
{
	Range< Dim > fRange;
	for( unsigned int d=0 ; d<Dim ; d++ ) fRange.first[d] = 0 , fRange.second[d] = res+1;
	return fRange;
}



template< unsigned int Dim >
typename ScalarFunctions< Dim >::template IntegrationStencil< double , 2 , 0 > ScalarFunctions< Dim >::MassStencil( unsigned int res )
{
	ScalarFunctions< Dim >::IntegrationStencil< double , 2 , 0 > stencil;

	Index< Dim > e;
	Range< Dim > r = Basis< Dim >::ElementSupport( e );

	auto f = [&]( Index< Dim > i1 , Index< Dim > i2 )
	{
		stencil._values[ Basis< Dim >::template _RelativeIndex<0>(e,i1) ][ Basis< Dim >::template _RelativeIndex<0>(e,i2) ] = ( ElementFunction< Dim >( e , i1 ) * ElementFunction< Dim >( e , i2 ) ).integral(res);
	};
	r.template process< 2 >( f );

	return stencil;
}

template< unsigned int Dim >
typename ScalarFunctions< Dim >::template IntegrationStencil< double , 2 , 0 > ScalarFunctions< Dim >::StiffnessStencil( unsigned int res )
{
	ScalarFunctions< Dim >::IntegrationStencil< double , 2 , 0 > stencil;

	Index< Dim > e;

	auto f = [&]( Index< Dim > i1 , Index< Dim > i2 )
	{
		ElementVector< Dim > v1 = ElementFunction< Dim >(e,i1).gradient() , v2 = ElementFunction< Dim >(e,i2).gradient();
		double integral = 0;
		for( unsigned int d=0 ; d<Dim ; d++ ) integral +=( v1[d] * v2[d] ).integral(res);
		stencil._values[ Basis< Dim >::template _RelativeIndex<0>(e,i1) ][ Basis< Dim >::template _RelativeIndex<0>(e,i2) ] = integral;
	};
	Basis< Dim >::ElementSupport( e ).template process< 2 >( f );

	return stencil;
}


template< unsigned int Dim >
typename ScalarFunctions< Dim >::template IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 2 , 0 > ScalarFunctions< Dim >::ConstantWeightedStiffnessStencil( unsigned int res )
{
	ScalarFunctions< Dim >::IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 2 , 0 > stencil;

	Index< Dim > e;
	auto f = [&]( Index< Dim > i1 , Index< Dim > i2 )
	{
		ElementVector< Dim > v1 = ElementFunction< Dim >(e,i1).gradient() , v2 = ElementFunction< Dim >(e,i2).gradient();
		MishaK::SquareMatrix< double , Dim > &m = stencil._values[ Basis< Dim >::template _RelativeIndex<0>(e,i1) ][ Basis< Dim >::template _RelativeIndex<0>(e,i2) ];
		for( unsigned int d1=0 ; d1<Dim ; d1++ ) for( unsigned int d2=0 ; d2<Dim ; d2++ ) m(d2,d1) += ( v1[d1] * v2[d2] ).integral(res);
	};
	Basis< Dim >::ElementSupport( e ).template process< 2 >( f );

	return stencil;
}

template< unsigned int Dim >
typename ScalarFunctions< Dim >::template IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 3 , 0 > ScalarFunctions< Dim >::LinearWeightedStiffnessStencil( unsigned int res )
{
	ScalarFunctions< Dim >::IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 3 , 0 > stencil;

	Index< Dim > e;
	auto f = [&]( Index< Dim > i1 , Index< Dim > i2 , Index< Dim > i3 )
		{
			ElementVector< Dim > v1 = ElementFunction< Dim >(e,i1).gradient() , v2 = ElementFunction< Dim >(e,i2).gradient();
			ElementFunction< Dim > f = ElementFunction< Dim >(e,i3);
			MishaK::SquareMatrix< double , Dim > &m = stencil._values[ Basis< Dim >::template _RelativeIndex<0>(e,i1) ][ Basis< Dim >::template _RelativeIndex<0>(e,i2) ][ Basis< Dim >::template _RelativeIndex<0>(e,i3) ];
			for( unsigned int d1=0 ; d1<Dim ; d1++ ) for( unsigned int d2=0 ; d2<Dim ; d2++ ) m(d2,d1) += ( v1[d1] * v2[d2] * f ).integral(res);
		};
	Basis< Dim >::ElementSupport( e ).template process< 3 >( f );

	return stencil;
}

template< unsigned int Dim >
typename ScalarFunctions< Dim >::template IntegrationStencil< double , 1 , 0 > ScalarFunctions< Dim >::ValueStencil( unsigned int res )
{
	ScalarFunctions< Dim >::IntegrationStencil< double , 1 , 0 > stencil;
	Index< Dim > e;

	auto f = [&]( Index< Dim > f )
	{
		ElementFunction< Dim > v(e,f);
		stencil._values[ Basis< Dim >::template _RelativeIndex<0>(e,f) ] = v.integral(res);
	};
	Basis< Dim >::ElementSupport( e ).process( f );
	return stencil;
}


template< unsigned int Dim >
typename ScalarFunctions< Dim >::template IntegrationStencil< Point< double , Dim > , 1 , 0 > ScalarFunctions< Dim >::PartialDerivativeStencil( unsigned int res )
{
	ScalarFunctions< Dim >::IntegrationStencil< Point< double , Dim > , 1 , 0 > stencil;
	Index< Dim > e;

	auto f = [&]( Index< Dim > f )
	{
		ElementVector< Dim > v = ElementFunction< Dim >(e,f).gradient();
		Point< double , Dim > &_v = stencil._values[ Basis< Dim >::template _RelativeIndex<0>(e,f) ];
		for( unsigned int d=0 ; d<Dim ; d++ ) _v[d] += v[d].integral(res);
	};
	Basis< Dim >::ElementSupport( e ).process( f );
	return stencil;
}

template< unsigned int Dim >
template< typename T , typename Indexer /* = Hat::BaseIndexer< Dim > */ >
T ScalarFunctions< Dim >::operator()( const Indexer & indexer , const IntegrationStencil< T , 2 , 0 > &stencil , const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	std::vector< double > integrals( ThreadPool::NumThreads() , 0 );
	ThreadPool::ParallelFor
	(
		0 , indexer.numElements() ,
		[&]( unsigned int t , size_t e )
		{
			Index< Dim > E = indexer.elementIndex( e );

			Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.efNeighbors( e , t );

			for( unsigned int i1=0 ; i1<Window::IsotropicSize< Dim , 2 >() ; i1++ ) if( neighbors.neighbors.data[i1]!=-1 )
			{
				unsigned int f1 = neighbors.neighbors.data[i1];
				Index< Dim > F1 = indexer.functionIndex( f1 );
				for( unsigned int i2=0 ; i2<Window::IsotropicSize< Dim , 2 >() ; i2++ ) if( neighbors.neighbors.data[i2]!=-1 )
				{
					unsigned int f2 = neighbors.neighbors.data[i2];
					Index< Dim > F2 = indexer.functionIndex( f2 );
					integrals[t] += stencil( E , F1 , F2 ) * x[f1] * y[f2];
				}
			}
		}
	);


	T integral{};
	for( unsigned int i=0 ; i<integrals.size() ; i++ ) integral += integrals[i];
	return integral;
}

template< unsigned int Dim >
template< typename T , typename Indexer /* = Hat::BaseIndexer< Dim > */ >
T ScalarFunctions< Dim >::operator()( const Indexer & indexer , const FullIntegrationStencil< T , 0 > &stencil , const Eigen::VectorXd &x , const Eigen::VectorXd &y ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;

	std::vector< double > integrals( ThreadPool::NumThreads() , 0 );
	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( unsigned int t , size_t f1 )
		{
			Index< Dim > F1 = indexer.functionIndex( f1 );
			Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , t );
			const typename Hat::ScalarFunctions< Dim >::FullIntegrationStencil< T , 0 >::Row &row = stencil.row( F1 );
			for( unsigned int j=0 ; j<row.size() ; j++ )
			{
				size_t f2 = neighbors.data[ std::get< 1 >( row[j] ) ];
				if( f2!=-1 ) integrals[t] += std::get< 2 >( row[j] ) * x[f1] * y[f2];
			}
		}
	);

	T integral{};
	for( unsigned int i=0 ; i<integrals.size() ; i++ ) integral += integrals[i];
	return integral;
}


template< unsigned int Dim >
template< typename T , typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::Matrix< T , 1 , Eigen::Dynamic > ScalarFunctions< Dim >::operator()( const Indexer & indexer , const IntegrationStencil< T , 2 , 0 > &stencil , const Eigen::VectorXd &x ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	Eigen::Matrix< T , 1 , Eigen::Dynamic > v = Eigen::Matrix< T , 1 , Eigen::Dynamic >::Zero( indexer.numFunctions() );

	for( int e=0 ; e<indexer.numElements() ; e++ )
	{
		Index< Dim > E = indexer.elementIndex( e );

		Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.efNeighbors( e , 0 );

		for( unsigned int i1=0 ; i1<Window::IsotropicSize< Dim , 2 >() ; i1++ ) if( neighbors.neighbors.data[i1]!=-1 )
		{
			unsigned int f1 = neighbors.neighbors.data[i1];
			Index< Dim > F1 = indexer.functionIndex( f1 );
			for( unsigned int i2=0 ; i2<Window::IsotropicSize< Dim , 2 >() ; i2++ ) if( neighbors.neighbors.data[i2]!=-1 )
			{
				unsigned int f2 = neighbors.neighbors.data[i2];
				Index< Dim > F2 = indexer.functionIndex( f2 );
				v[f2] += stencil( E , f1 , f2 ) * x[f1];
			}
		}
	}

	return v;
}

template< unsigned int Dim >
template< typename T , typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::Matrix< T , 1 , Eigen::Dynamic > ScalarFunctions< Dim >::operator()( const Indexer & indexer , const FullIntegrationStencil< T , 0 > &stencil , const Eigen::VectorXd &x ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;

	Eigen::Matrix< T , 1 , Eigen::Dynamic > v = Eigen::Matrix< T , 1 , Eigen::Dynamic >::Zero( indexer.numFunctions() );

	for( int f1=0 ; f1<indexer.numFunctions() ; f1++ )
	{
		Index< Dim > F1 = indexer.functionIndex( f1 );

		Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , 0 );

		const typename Hat::ScalarFunctions< Dim >::FullIntegrationStencil< T , 0 >::Row &row = stencil.row( F1 );

		for( unsigned int j=0 ; j<row.size() ; j++ )
		{
			Index< Dim > I = std::get<0>(row[j]) + Off;
			size_t f2 = neighbors( &I[0] );
			if( f2!=-1 ) v[f2] += std::get<2>(row[j]) * x[f1];
		}
	}
	return v;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::systemMatrix( const Indexer & indexer , IntegrationStencil< double , 2 , 0 > stencil ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	static const unsigned int Radius = 0;
	Hat::Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;

	// If a function is supported on a radius of (1+Radius) cells, then two functions have overlapping support if
	// they are in each other's (1+2*Radius)-ring
	MatrixInfo< 1+2*Radius , true > matrixInfo( _r );
	Eigen::SparseMatrix< double > M( indexer.numFunctions() , indexer.numFunctions() );
	Eigen::VectorXi rowSizes( (int)indexer.numFunctions() );
	ThreadPool::ParallelFor( 0 , indexer.numFunctions() , [&]( size_t i ){ Index< Dim > I = indexer.functionIndex( (size_t)i ) ; rowSizes[i] = (int)matrixInfo.entries(I,true); } );
	M.reserve( rowSizes );

	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( unsigned int t , size_t f1 )
		{
			Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , 0 );
			Index< Dim > F1 = indexer.functionIndex( f1 );

			auto Kernel = [&]( Index< Dim > F2 , size_t e , bool flip )
				{
					Hat::Index< Dim > I = F2-F1+Off;
					size_t f2 = neighbors( &I[0] );
					if( f2!=-1 )
					{
						Range< Dim > range = Range< Dim >::Intersect( Basis< Dim >::FunctionSupport( F1 ) , Basis< Dim >::FunctionSupport( F2 ) , _eRange );
						M.insert( (int)f2 , (int)f1 ) = stencil( range , F1 , F2 );
					}
				};
			matrixInfo.processAll( F1 , Kernel );
		}
	);

	M.makeCompressed();
	return M;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::systemMatrix( const Indexer & indexer , FullIntegrationStencil< double , 0 > stencil ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	static const unsigned int Radius = 0;
	Hat::Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;

	Eigen::SparseMatrix< double > M( indexer.numFunctions() , indexer.numFunctions() );
	Eigen::VectorXi rowSizes( (int)indexer.numFunctions() );
	ThreadPool::ParallelFor( 0 , indexer.numFunctions() , [&]( size_t i ){ rowSizes[i] = (int)stencil.row( indexer.functionIndex( (size_t)i ) ).size(); } );

	M.reserve( rowSizes );

	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( size_t f1 )
		{
			Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , 0 );
			Index< Dim > F1 = indexer.functionIndex( f1 );

			const typename Hat::ScalarFunctions< Dim >::FullIntegrationStencil< double , Radius >::Row &row = stencil.row( F1 );
			for( unsigned int j=0 ; j<row.size() ; j++ )
			{
				Hat::Index< Dim > I = std::get<0>(row[j]) + Off;
				size_t f2 = neighbors( &I[0] );
				if( f2!=-1 ) M.insert( (int)f2 , (int)f1 ) = std::get<2>(row[j]);
			}
		}
	);

	M.makeCompressed();
	return M;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::boundarySystemMatrix( const Indexer & indexer , typename ScalarFunctions< Dim-1 >::template IntegrationStencil< double , 2 , 0 > stencil ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	static const unsigned int Radius = 0;
	Hat::Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;

	if constexpr( Dim==1 )
	{
		Eigen::SparseMatrix< double > M( indexer.numFunctions() , indexer.numFunctions() );
		return M;
	}
	else
	{
		auto FullIndex = [&]( Index< Dim-1 > _F , unsigned int d , unsigned int offset )
			{
				Index< Dim > F;
				for( unsigned int _d=0 ; _d<d ; _d++ ) F[_d] = _F[_d];
				F[d] = offset;
				for( unsigned int _d=d+1 ; _d<Dim ; _d++ ) F[_d] = _F[_d-1];
				return F;
			};

		// Not parallelized, but handles multiple inserts of the same coefficients
		ScalarFunctions< Dim-1 > _scalars( _r );
		typename ScalarFunctions< Dim-1 >::template FullIntegrationStencil< double , Radius > fullStencil( stencil , _r );

		std::vector< Eigen::Triplet< double > > triplets;

		auto AddEntries = [&]( size_t f1 , Hat::Index< Dim > F1 , unsigned int d , unsigned int offset , Window::IsotropicStaticWindow< size_t , Dim , 3 > &neighbors )
			{
				Hat::Index< Dim-1 > _F1;
				for( unsigned int dd=0 , _d=0 ; dd<Dim ; dd++ ) if( dd!=d ) _F1[_d++] = F1[dd];

				const typename Hat::ScalarFunctions< Dim-1 >::template FullIntegrationStencil< double , Radius >::Row &row = fullStencil.row( _F1 );
				for( unsigned int j=0 ; j<row.size() ; j++ )
				{
					Index< Dim-1 > _F2 = _F1 + std::get<0>( row[j] );
					Index< Dim > F2 = FullIndex( _F2 , d , offset );
					Index< Dim > I = F2 - F1 + Off;
					size_t f2 = neighbors( &I[0] );
					if( f2!=-1 )
					{
						if( f1<=f2 )
						{
							double integral = std::get<2>( row[j] );
							if( f1==f2 ) integral /= 2;
							triplets.push_back( Eigen::Triplet< double >( (int)f1 , (int)f2 , integral) );
						}
					}
				}
			};

		for( size_t f1=0 ; f1<indexer.numFunctions() ; f1++ )
		{
			Hat::Index< Dim > F1 = indexer.functionIndex( f1 );

			bool onBoundary = false;
			for( unsigned int d=0 ; d<Dim ; d++ ) onBoundary |= F1[d]==0 || F1[d]==_r;

			if( onBoundary )
			{
				Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , 0 );

				for( unsigned int d=0 ; d<Dim ; d++ )
				{
					if( F1[d]== 0 ) AddEntries( f1 , F1 , d ,  0 , neighbors );
					if( F1[d]==_r ) AddEntries( f1 , F1 , d , _r , neighbors );
				}
			}
		}

		{
			Eigen::SparseMatrix< double > M( indexer.numFunctions() , indexer.numFunctions() );
			M.setFromTriplets( triplets.begin() , triplets.end() );
			triplets.clear();
			Eigen::SparseMatrix< double > Mt = M.transpose();
			return M + Mt;
		}
	}
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::deltaMass( const Indexer & indexer , SampleFunctor && F , size_t sampleNum , WeightFunctor && wF ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	// The values of the corner functions at the samples
	struct FunctionValues
	{
		double values[1<<Dim];
		FunctionValues( Point< double , Dim > p )
		{
			Index< Dim > e;
			double _values[2][Dim];
			for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Dim ; d++ ) _values[i][d] = i==0 ? (1.-p[d]) : p[d];

			auto f = [&]( Index< Dim > f )
				{
					double value = 1.;
					for( unsigned int d=0 ; d<Dim ; d++ ) value *= f[d]==e[d] ? _values[0][d] : _values[1][d];
					values[ Basis< Dim >::template _RelativeIndex<0>( e , f ) ] = value;
				};
			Basis< Dim >::ElementSupport( e ).process( f );
		}
	};

	MultiThreadedTriplets< double > triplets;

	OrderedSamples< Dim , double > orderedSamples( [&]( size_t i ){ Point< double , Dim > p = F(i) ; return std::pair< Point< double , Dim > , double >( p , wF(p) ); } , sampleNum , _r );

	ThreadPool::ParallelFor
	(
		0 , orderedSamples.size() ,
		[&]( unsigned int t , size_t i )
		{
			Hat::Index< Dim > E = orderedSamples[i].first;
			const std::vector< std::pair< Point< double , Dim > , double > > &subSamples = orderedSamples[i].second;

			Point< double , Dim > p;
			for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d] + 0.5 ) / _r;
			Window::IsotropicStaticWindow< size_t , Dim , 2 > nbrs = indexer.fNeighbors( p , t );
			// Get the values of each of the function corner functions at each of the samples
			std::vector< FunctionValues > functionValues;
			functionValues.reserve( subSamples.size() );
			for( unsigned int j=0 ; j<subSamples.size() ; j++ )
			{
				Point< double , Dim > p = subSamples[j].first * _r;
				for( unsigned int d=0 ; d<Dim ; d++ ) p[d] -= E[d];
				functionValues.emplace_back( p );
			}

			for( unsigned int i1=0 ; i1<Window::IsotropicSize< Dim , 2 >() ; i1++ )
			{
				size_t idx1 = nbrs.data[i1];
				if( idx1!=-1 )
				{
					Index< Dim > F1 = indexer.functionIndex( idx1 );
					for( unsigned int i2=0 ; i2<Window::IsotropicSize< Dim , 2 >() ; i2++ )
					{
						size_t idx2 = nbrs.data[i2];
						if( idx2!=-1 )
						{
							Index< Dim > F2 = indexer.functionIndex( idx2 );
							if( F1<=F2 )
							{
								double value = 0;
								for( unsigned int j=0 ; j<subSamples.size() ; j++ ) value += functionValues[j].values[ Basis< Dim >::template _RelativeIndex<0>(E,F1) ] * functionValues[j].values[ Basis< Dim >::template _RelativeIndex<0>(E,F2) ] * subSamples[j].second;
								if( F1==F2 ) value /= 2.;
								triplets[t].push_back( Eigen::Triplet< double >( (int)idx1 , (int)idx2 , value ) );
							}
						}
					}
				}
			}
		}
	);

	Eigen::SparseMatrix< double > E( indexer.numFunctions(), indexer.numFunctions());
	E.setFromTriplets( triplets.begin() , triplets.end() );
	Eigen::SparseMatrix< double > Et = E.transpose();
	return E + Et;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::deltaMass( const Indexer & indexer , SampleFunctor && F , const OrderedSampler< Dim > &orderedSampler , WeightFunctor && wF ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );
	static_assert( std::is_convertible_v< SampleFunctor , std::function< Point< double , Dim > ( size_t ) > > , "[ERROR] SampleFunctor poorly formed" );
	static_assert( std::is_convertible_v< WeightFunctor , std::function< double ( Point< double , Dim > , unsigned int ) > > , "[ERROR] WeightFunctor poorly formed" );

	unsigned int shift = 0;
	if( _r>orderedSampler.res() ) MK_ERROR_OUT( "Ordered sampler insufficiently refined" );
	while( (_r<<shift)<orderedSampler.res() ) shift++;
	if( (_r<<shift)!=orderedSampler.res() ) MK_ERROR_OUT( "Resolutions are not power of two multiples" );

	// The values of the corner functions at the samples
	struct FunctionValues
	{
		double values[1<<Dim];
		FunctionValues( Point< double , Dim > p )
		{
			Index< Dim > e;
			double _values[2][Dim];
			for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Dim ; d++ ) _values[i][d] = i==0 ? (1.-p[d]) : p[d];

			auto f = [&]( Index< Dim > f )
				{
					double value = 1.;
					for( unsigned int d=0 ; d<Dim ; d++ ) value *= f[d]==e[d] ? _values[0][d] : _values[1][d];
					values[ Basis< Dim >::template _RelativeIndex<0>( e , f ) ] = value;
				};
			Basis< Dim >::ElementSupport( e ).process( f );
		}
	};

	MultiThreadedTriplets< double > triplets;

	ThreadPool::ParallelFor
	(
		0 , orderedSampler.size() ,
		[&]( unsigned int t , size_t i )
		{
			Hat::Index< Dim > E = orderedSampler[i].first;
			for( unsigned int d=0 ; d<Dim ; d++ ) E[d] >>= shift;
			const std::vector< size_t > &subSampleIndices = orderedSampler[i].second;
			std::vector< std::pair< Point< double , Dim > , double > > subSamples( subSampleIndices.size() );
			bool hasWeight = false;
			for( size_t j=0 ; j<subSampleIndices.size() ; j++ )
			{
				subSamples[j].first = F( subSampleIndices[j] );
				subSamples[j].second = wF( subSamples[j].first , t );
				if( subSamples[j].second ) hasWeight = true;
			}
			if( hasWeight )
			{
				Point< double , Dim > p;
				for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d] + 0.5 ) / _r;
				Window::IsotropicStaticWindow< size_t , Dim , 2 > nbrs = indexer.fNeighbors( p , t );
				// Get the values of each of the function corner functions at each of the samples
				std::vector< FunctionValues > functionValues;
				functionValues.reserve( subSamples.size() );
				for( unsigned int j=0 ; j<subSamples.size() ; j++ )
				{
					Point< double , Dim > p = subSamples[j].first * _r;
					for( unsigned int d=0 ; d<Dim ; d++ ) p[d] -= E[d];
					functionValues.emplace_back( p );
				}

				for( unsigned int i1=0 ; i1<Window::IsotropicSize< Dim , 2 >() ; i1++ )
				{
					size_t idx1 = nbrs.data[i1];
					if( idx1!=-1 )
					{
						Index< Dim > F1 = indexer.functionIndex( idx1 );
						for( unsigned int i2=0 ; i2<Window::IsotropicSize< Dim , 2 >() ; i2++ )
						{
							size_t idx2 = nbrs.data[i2];
							if( idx2!=-1 )
							{
								Index< Dim > F2 = indexer.functionIndex( idx2 );
								if( F1<=F2 )
								{
									double value = 0;
									for( unsigned int j=0 ; j<subSamples.size() ; j++ ) value += functionValues[j].values[ Basis< Dim >::template _RelativeIndex<0>(E,F1) ] * functionValues[j].values[ Basis< Dim >::template _RelativeIndex<0>(E,F2) ] * subSamples[j].second;
									if( F1==F2 ) value /= 2.;
									triplets[t].push_back( Eigen::Triplet< double >( (int)idx1 , (int)idx2 , value ) );
								}
							}
						}
					}
				}
			}
		}
	);

	Eigen::SparseMatrix< double > E( indexer.numFunctions() , indexer.numFunctions());
	E.setFromTriplets( triplets.begin() , triplets.end() );
	Eigen::SparseMatrix< double > Et = E.transpose();
	return E + Et;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::deltaMass( const Indexer & indexer , const std::vector< Point< double , Dim > > &samples , WeightFunctor && wF ) const
{
	return deltaMass( std::forward< Indexer >( indexer ) , [&]( size_t idx ){ return samples[idx]; } , samples.size() , std::forward< WeightFunctor >( wF ) );
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::deltaMass( const Indexer & indexer , SampleFunctor && F , size_t sampleNum ) const
{
	return deltaMass( std::forward< Indexer >( indexer ) , std::forward< SampleFunctor >( F ) , sampleNum , []( Point< double , Dim > ){ return 1.;} );
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx , unsigned int ) > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::deltaMass( const Indexer & indexer , SampleFunctor && F , const OrderedSampler< Dim > &orderedSampler ) const
{
	return deltaMass( std::forward< Indexer >( indexer ) , std::forward< SampleFunctor >( F ) , orderedSampler , []( Point< double , Dim > , unsigned int ){ return 1.;} );
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::deltaMass( const Indexer & indexer , const std::vector< Point< double , Dim > > &samples ) const
{
	return deltaMass( std::forward< Indexer >( indexer ) , samples , []( Point< double , Dim > ){ return 1.;} );
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::boundaryMass( const Indexer &indexer ) const
{
	if constexpr( Dim==0 )
	{
		MK_ERROR_OUT( "Cannot compute boundary for dimension zero" );
		return Eigen::SparseMatrix< double >( indexer.numFunctions() , indexer.numFunctions() );
	}
	else
	{
		typename ScalarFunctions< Dim-1 >::template IntegrationStencil< double , 2 , 0 > stencil = ScalarFunctions< Dim-1 >::MassStencil( _r );
		return boundarySystemMatrix( indexer , stencil );
	}
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::boundaryStiffness( const Indexer &indexer ) const
{
	if constexpr( Dim==0 || Dim==1 )
	{
		MK_ERROR_OUT( "Cannot compute boundary stiffness for dimension zero or one" );
		return Eigen::SparseMatrix< double >( indexer.numFunctions() , indexer.numFunctions() );
	}
	else
	{
		typename ScalarFunctions< Dim-1 >::template IntegrationStencil< double , 2 , 0 > stencil = ScalarFunctions< Dim-1 >::StiffnessStencil( _r );
		return boundarySystemMatrix( indexer , stencil );
	}
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
double ScalarFunctions< Dim >::value( const Indexer & indexer , const Eigen::VectorXd &x , Point< double , Dim > p , unsigned int thread ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.fNeighbors( p , thread );

	p *= _r;
	Index< Dim > E;
	for( unsigned int d=0 ; d<Dim ; d++ ) E[d] = (int)floor( p[d] );
	for( unsigned int d=0 ; d<Dim ; d++ ) if( E[d]==_r ) E[d]--;
	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] -= E[d];

	double values[2][Dim];
	for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Dim ; d++ ) values[i][d] = i==0 ? (1.-p[d]) : p[d];

	double value = 0;
	auto Kernel = [&]( Index< Dim > dF )
		{
			size_t f = neighbors( &dF[0] );
			if( f!=-1 )
			{
				double v = x[f];
				for( unsigned int d=0 ; d<Dim ; d++ ) v *= dF[d]==0 ? values[0][d] : values[1][d];
				value += v;
			}
		};
	Basis< Dim >::ElementSupport( Index< Dim >() ).process( Kernel );
	return value;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Point< double , Dim > ScalarFunctions< Dim >::gradient( const Indexer & indexer , const Eigen::VectorXd &x , Point< double , Dim > p , unsigned int thread ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.fNeighbors( p , thread );

	p *= _r;
	Index< Dim > E;
	for( unsigned int d=0 ; d<Dim ; d++ ) E[d] = (int)floor( p[d] );
	for( unsigned int d=0 ; d<Dim ; d++ ) if( E[d]==_r ) E[d]--;
	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] -= E[d];

	double values[2][Dim] , dValues[2][Dim];
	for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Dim ; d++ )
	{
		values[i][d] = i==0 ? (1.-p[d]) : p[d];
		dValues[i][d] = i==0 ? -(int)_r : (int)_r;
	}

	Point< double , Dim > gradient;
	for( unsigned int d=0 ; d<Dim ; d++ )
	{
		auto Kernel = [&]( Index< Dim > dF )
			{
				size_t f = neighbors( &dF[0] );
				if( f!=-1 )
				{
					double v = x[f];
					for( unsigned int dd=0 ; dd<Dim ; dd++ )
						if( d!=dd ) v *= dF[dd]==0 ?  values[0][dd] :  values[1][dd];
						else        v *= dF[dd]==0 ? dValues[0][dd] : dValues[1][dd];
					gradient[d] += v;
				}
			};
		Basis< Dim >::ElementSupport( Index< Dim >() ).process( Kernel );
	}
	return gradient;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ , typename PiecewiseConstantScalarField /* = std::function< double ( Hat::Index< Dim > E ) > */ >
Eigen::VectorXd ScalarFunctions< Dim >::valueDual( const Indexer &indexer , PiecewiseConstantScalarField SF ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );
	static_assert( std::is_convertible_v< PiecewiseConstantScalarField , std::function< double ( Hat::Index< Dim > ) > > , "[ERROR] PiecewiseConstantScalarField poorly formed" );

	IntegrationStencil< double , 1 , 0 >  stencil = ValueStencil( _r );

	Eigen::VectorXd d( indexer.numFunctions() );

	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( unsigned int t , size_t f )
		{
			Index< Dim > F = indexer.functionIndex( f );
			Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.feNeighbors( f , t );

			double value = 0;
			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
			{
				size_t e = neighbors.data[i];
				if( e!=-1 )
				{
					Index< Dim > E = indexer.elementIndex( e );
					value += stencil( E , F ) * SF( E );
				}
			}
			d[f] = value;
		}
	);

	return d;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ >
Eigen::VectorXd ScalarFunctions< Dim >::valueDual( const Indexer & indexer , const Eigen::VectorXd &x ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	IntegrationStencil< double , 2 , 0 >  stencil = MassStencil( _r );

	Eigen::VectorXd d( indexer.numFunctions() );
	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( unsigned int t , size_t f1 )
		{
			Index< Dim > F1 = indexer.functionIndex( f1 );
			Window::IsotropicStaticWindow< size_t , Dim , 2 > eNeighbors = indexer.feNeighbors( f1 , t );

			double value = 0;

			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
			{
				size_t e = eNeighbors.data[i];
				if( e!=-1 )
				{
					Index< Dim > E = indexer.elementIndex( e );
					Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors = indexer.efNeighbors( e , t );
					for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
					{
						size_t f2 = fNeighbors.data[i];
						if( f2!=-1 )
						{
							Index< Dim > F2 = indexer.functionIndex( f2 );
							value += stencil( E , F1 , F2 ) * x[f2];
						}
					}
				}
			}

			d[f1] = value;
		}
	);

	return d;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ , typename PiecewiseConstantVectorField /* = std::function< Point< double , Dim > ( Hat::Index< Dim > E ) > */ >
Eigen::VectorXd ScalarFunctions< Dim >::gradientDual( const Indexer & indexer , PiecewiseConstantVectorField VF ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );
	static_assert( std::is_convertible_v< PiecewiseConstantVectorField , std::function< Point< double , Dim > ( Hat::Index< Dim > ) > > , "[ERROR] PiecewiseConstantVectorField poorly formed" );

	IntegrationStencil< Point< double , Dim > , 1 , 0 >  stencil = PartialDerivativeStencil( _r );

	Eigen::VectorXd d( indexer.numFunctions() );

	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( unsigned int t , size_t f )
		{
			Index< Dim > F = functionIndex( f );

			Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.feNeighbors( f , t );

			double value = 0;
			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
			{
				size_t e = neighbors.data[i];
				if( e!=-1 )
				{
					Hat::Index< Dim > E = indexer.elementIndex( e );
					value += Point< double , Dim >::Dot( stencil( E , F ) , VF( E ) );
				}
			}
			d[f] = value;
		}
	);

	return d;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ >
Eigen::VectorXd ScalarFunctions< Dim >::gradientDual( const Indexer & indexer , const Eigen::VectorXd &x ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	IntegrationStencil< double , 2 , 0 >  stencil = StiffnessStencil( _r );

	Eigen::VectorXd d( indexer.numFunctions() );

	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() , 
		[&]( unsigned int t , size_t f1 )
		{
			Index< Dim > F1 = indexer.functionIndex( f1 );
			Window::IsotropicStaticWindow< size_t , Dim , 2 > eNeighbors = indexer.feNeighbors( f1 , t );

			double value = 0;

			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
			{
				size_t e = eNeighbors.data[i];
				if( e!=-1 )
				{
					Index< Dim > E = indexer.elementIndex( e );
					Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors = indexer.efNeighbors( e , t );
					for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
					{
						size_t f2 = fNeighbors.data[i];
						if( f2!=-1 )
						{
							Index< Dim > F2 = indexer.functionIndex( f2 );
							value += stencil( E , F1 , F2 ) * x[f2];
						}
					}
				}
			}

			d[f1] = value;
		}
	);

	return d;
}

template< unsigned int Dim >
template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , Data > ( unsigned int idx ) > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
std::vector< Data > ScalarFunctions< Dim >::deltaDual( const Indexer &indexer , SampleFunctor F , size_t sampleNum , WeightFunctor wF ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );

	std::vector< Data > b( indexer.numFunctions() );
	for( unsigned int i=0 ; i<indexer.numFunctions() ; i++ ) b[i] = {};

	// The values of the corner functions at the samples
	struct FunctionValues
	{
		double values[1<<Dim];
		FunctionValues( void ){}

		FunctionValues( Point< double , Dim > p )
		{
			Index< Dim > e;
			double _values[2][Dim];
			for( unsigned int i=0 ; i<2 ; i++ ) for( unsigned int d=0 ; d<Dim ; d++ ) _values[i][d] = i==0 ? (1.-p[d]) : p[d];

			auto f = [&]( Index< Dim > f )
				{
					double value = 1.;
					for( unsigned int d=0 ; d<Dim ; d++ ) value *= f[d]==e[d] ? _values[0][d] : _values[1][d];
					values[ Basis< Dim >::template _RelativeIndex<0>( e , f ) ] = value;
				};
			Basis< Dim >::ElementSupport( e ).process( f );
		}
	};

	struct IndexedSample
	{
		Point< double , Dim > position;
		Data data;
		size_t index;
		double weight;
		IndexedSample( Point< double , Dim > p , Data d , double w , unsigned int res ) : position(p) , data(d) , weight(w) , index(0)
		{
			static const unsigned int NumBits = std::min< unsigned int >( ( sizeof(size_t)*8 ) / Dim , 16 );
			static const size_t MaxRes = (size_t)1<<NumBits;
			if( res>MaxRes ) MK_ERROR_OUT( "Resolution too large: " , res , " >= " , MaxRes , " " , NumBits );
			Index< Dim > idx;
			for( unsigned int d=0 ; d<Dim ; d++ )
			{
				if( position[d]<0 || position[d]>=1 ) MK_ERROR_OUT( "Sample out bounds: " , p );
				idx[d] = (unsigned int)floor( position[d] * res );
			}
			for( unsigned int i=0 ; i<NumBits ; i++ ) for( unsigned int d=0 ; d<Dim ; d++ ) index |= ( (idx[d]>>i)&1 ) << ( i*Dim + d );
		}
	};

	std::vector< IndexedSample > samples;
	samples.reserve( sampleNum );
	for( unsigned int i=0 ; i<sampleNum ; i++ )
	{
		std::pair< Point< double , Dim > , Data > sample = F(i);
		double w = wF( sample.first );
		if( w>0 ) samples.emplace_back( sample.first , sample.second , w , _r );
	}
	std::sort( samples.begin() , samples.end() , []( const IndexedSample &i1 , const IndexedSample &i2 ){ return i1.index<i2.index; } );

	std::vector< FunctionValues > functionValues;
	unsigned int start=0;
	while( start<samples.size() )
	{
		// compute the [start,end) range for samples mapping to the same cell
		unsigned int end;
		for( end=start ; end<samples.size() && samples[end].index==samples[start].index ; end++ );

		auto SampleCell = [&]( Point< double , Dim > p )
			{
				Index< Dim > E;
				p *= _r;
				for( unsigned int d=0 ; d<Dim ; d++ ) E[d] = (int)floor( p[d] );
				for( unsigned int d=0 ; d<Dim ; d++ ) if( E[d]==_r ) E[d]--;
				return E;
			};

		// Get the cell using one of the samples (e.g. the first)
		Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.fNeighbors( samples[start].position , 0 );
		Index< Dim > E = SampleCell( samples[start].position );

		// Get the values of each of the function corner functions at each of the samples
		functionValues.resize(0);
		functionValues.reserve( end-start );
		for( unsigned int i=start ; i<end ; i++ )
		{
			Point< double , Dim > p = samples[i].position * _r;
			for( unsigned int d=0 ; d<Dim ; d++ ) p[d] -= E[d];
			functionValues.emplace_back( p );
		}

		for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
		{
			size_t f = neighbors.data[i];
			if( f!=-1 )
			{
				Hat::Index< Dim > F = indexer.functionIndex( f );
				Data data = {};
				for( unsigned int i=start ; i<end ; i++ ) data += samples[i].data * functionValues[i-start].values[ Basis< Dim >::template _RelativeIndex<0>( E , F ) ] * samples[i].weight;
				b[f] += data;
			}
		}

		start = end;
	}

	return b;
}

template< unsigned int Dim >
template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ , typename WeightFunctor /* = std::function< double ( Point< double , Dim > ) */ >
std::vector< Data > ScalarFunctions< Dim >::deltaDual( const Indexer &indexer , const std::vector< std::pair< Point< double , Dim > , Data > > &samples , WeightFunctor wF ) const
{
	return deltaDual< Data >( indexer , [&]( size_t idx ){ return samples[idx]; } , samples.size() , wF );
}

template< unsigned int Dim >
template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ , typename SampleFunctor /* = std::function< std::pair< Point< double , Dim > , Data > ( unsigned int idx ) > */ >
std::vector< Data > ScalarFunctions< Dim >::deltaDual( const Indexer &indexer , SampleFunctor F , size_t sampleNum ) const
{
	return deltaDual< Data >( indexer , F , sampleNum , []( Point< double , Dim > ){ return 1.;} );
}

template< unsigned int Dim >
template< typename Data , typename Indexer /* = Hat::Indexer< Dim > */ >
std::vector< Data > ScalarFunctions< Dim >::deltaDual( const Indexer &indexer , const std::vector< std::pair< Point< double , Dim > , Data > > &samples ) const
{
	return deltaDual< Data >( indexer , samples , []( Point< double , Dim > ){ return 1.;} );
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::BaseIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::mass( const Indexer & indexer ) const { return systemMatrix( indexer , MassStencil(_r) ); }

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ , typename MassFunctor /* = std::function< double ( Index< Dim > E ) > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::mass( const Indexer & indexer , MassFunctor && mf ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );
	static_assert( std::is_convertible_v< MassFunctor , std::function< double ( Index< Dim > ) > > , "[ERROR] Poorly formed MassFunctor" );

	Hat::Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) Off[d] = 1;
	Range< Dim > eRange;
	for( unsigned int d=0 ; d<Dim ; d++ ) eRange.first[d] = 0 , eRange.second[d] = _r;
	IntegrationStencil< double , 2 , 0 > stencil = MassStencil( _r );

	// Determine which functions are supported on some cell
	std::vector< bool > supportedFunctions( indexer.numFunctions() , false );

	// [WARNING] Don't parallelize because of write-on-write conflict
	for( size_t e=0 ; e<indexer.numElements() ; e++ )
	{
		Hat::Index< Dim > E = indexer.elementIndex(e);
		if( mf( E )!=0 )
		{
			Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.efNeighbors( e , 0 );
			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
			{
				size_t f = neighbors.data[i];
				if( f!=-1 ) supportedFunctions[f] = true;
			}
		}
	}

	Eigen::SparseMatrix< double > S( indexer.numFunctions() , indexer.numFunctions() );

	MatrixInfo< 1 , true > matrixInfo( _r );

	Eigen::VectorXi rowSizes( (int)indexer.numFunctions() );
	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( size_t f )
		{
			Index< Dim > F = indexer.functionIndex(f);
			rowSizes[f] = supportedFunctions[f] ? (int)matrixInfo.entries(F,true) : 0;
		}
	);
	S.reserve( rowSizes );

	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( unsigned int t , size_t f1 )
		{
			Index< Dim > F1 = indexer.functionIndex( f1 );

			if( supportedFunctions[f1] )
			{
				Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = indexer.ffNeighbors( f1 , 0 );

				auto Kernel = [&]( Index< Dim > F2 , size_t e , bool flip )
					{
						Hat::Index< Dim > I = F2-F1+Off;
						size_t f2 = neighbors( &I[0] );
						double dot = 0;
						if( f2!=-1 )
						{
							auto Kernel = [&]( Index< Dim > E ){ dot += stencil( E , F1 , F2 ) * mf( E ); };
							// Iterate over all cells supported by the i-th function
							Range< Dim >::Intersect( Basis< Dim >::FunctionSupport( F1 ) , Basis< Dim >::FunctionSupport( F2 ) , eRange ).process( Kernel );
						}

						if( dot ) S.insert( (int)f2 , (int)f1 ) = dot;
					};
				matrixInfo.processAll( F1 , Kernel );
			}
		}
	);

	S.makeCompressed();
	return S;
}

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::stiffness( const Indexer & indexer ) const { return systemMatrix( indexer , StiffnessStencil(_r) ); }

template< unsigned int Dim >
template< typename Indexer /* = Hat::Indexer< Dim > */ , typename InnerProductFunctor /* = std::function< MishaK::SquareMatrix< double , Dim > ( Index< Dim > E [ , unsigned int t [ ) > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::stiffness( const Indexer & indexer , InnerProductFunctor && IPF , bool piecewiseLinear ) const
{
	static_assert( std::is_base_of_v< Hat::BaseIndexer< Dim > , Indexer > , "[ERROR] Indexer poorly formed" );
	static_assert( std::is_convertible_v< InnerProductFunctor , std::function< MishaK::SquareMatrix< double , Dim > ( Index< Dim > ) > > || std::is_convertible_v< InnerProductFunctor , std::function< MishaK::SquareMatrix< double , Dim > ( Index< Dim > , unsigned int ) > > , "[ERROR] InnerProductFunctor poorly formed" );

	static const bool NeedsThreadID = std::is_convertible_v< InnerProductFunctor , std::function< MishaK::SquareMatrix< double , Dim > ( Index< Dim > , unsigned int ) > >;
	Hat::Index< Dim > Off;
	Range< Dim > eRange;
	for( unsigned int d=0 ; d<Dim ; d++ ) eRange.first[d] = 0, eRange.second[d] = _r , Off[d] = 1;

	// Determine which functions are supported on a cell where the inner-product functor is non-zero
	std::vector< char > supportedFunctions( indexer.numFunctions() , 0 );
	ThreadPool::ParallelFor
	(
		0 , indexer.numElements() ,
		[&]( unsigned int t , size_t e )
		{
			Hat::Index< Dim > E = indexer.elementIndex(e);
			bool supported;
			if constexpr( NeedsThreadID ) supported = IPF( E ,t ).squareNorm() != 0;
			else                          supported = IPF( E ).squareNorm() != 0;
			if( supported )
			{
				Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors = indexer.efNeighbors( e , t );
				for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
				{
					size_t f = neighbors.data[i];
					if( f!=-1 ) SetAtomic( supportedFunctions[f] , (char)1 );
				}
			}
		}
	);

	Eigen::SparseMatrix< double > S( indexer.numFunctions() , indexer.numFunctions() );

	MatrixInfo< 1 , true > matrixInfo(_r);

	Eigen::VectorXi rowSizes((int)indexer.numFunctions());
	ThreadPool::ParallelFor
	(
		0 , indexer.numFunctions() ,
		[&]( size_t f )
		{
			Index< Dim > F = indexer.functionIndex(f);
			rowSizes[f] = supportedFunctions[f] ? (int)matrixInfo.entries( F , true ) : 0;
		}
	);
	S.reserve( rowSizes );

	if( piecewiseLinear )
	{
		IntegrationStencil< MishaK::SquareMatrix< double , Dim > , 3 , 0 > stencil = LinearWeightedStiffnessStencil( _r );

		ThreadPool::ParallelFor
		(
			0 , indexer.numFunctions() ,
			[&]( unsigned int t , size_t f1 )
			{
				Index< Dim > F1 = indexer.functionIndex(f1);

				if( supportedFunctions[f1] )
				{
					Window::IsotropicStaticWindow< size_t , Dim , 3 > fNeighbors = indexer.ffNeighbors( f1 , t );
					Window::IsotropicStaticWindow< size_t , Dim , 2 > eNeighbors = indexer.feNeighbors( f1 , t );

					// Precompute the values on the elements supported by F1
					Window::IsotropicStaticWindow< MishaK::SquareMatrix< double , Dim > , Dim , 3 >  ipf;
					{
						auto Kernel = [&]( Hat::Index< Dim > F )
							{
								Hat::Index< Dim > I = F - F1 + Off;
								if( fNeighbors( &I[0] )!=-1 )
									if constexpr( NeedsThreadID ) ipf( &I[0] ) = IPF( F , t );
									else                          ipf( &I[0] ) = IPF( F );
							};
						Range< Dim >::Intersect( Range< Dim >( F1 ).dilate(1) , eRange ).process( Kernel );
					}

					auto Kernel = [&]( Index< Dim > F2 , size_t e , bool flip )
						{
							Hat::Index< Dim > I = F2-F1+Off;
							size_t f2 = fNeighbors( &I[0] );
							if( f2!=-1 && supportedFunctions[f2] )
							{
								double dot = 0;
								auto _Kernel = [&]( Index< Dim > E )
									{
										Hat::Index< Dim > I = E - F1 + Off;
										if( eNeighbors( &I[0] )!=-1 )
										{
											unsigned int i1 = Basis< Dim >::template _RelativeIndex< 0 >( E , F1 ) , i2 = Basis< Dim >::template _RelativeIndex< 0 >( E , F2 );
											const SquareStencil< MishaK::SquareMatrix< double , Dim > , 1 , StencilSize< 0 >() > &_stencil = stencil[i1][i2];

											auto __Kernel = [&]( Index< Dim > F )
												{
													Hat::Index< Dim > I = F - F1 + Off;
													MishaK::SquareMatrix< double , Dim > m = ipf( &I[0] );
													unsigned int i = Basis< Dim >::template _RelativeIndex< 0 >( E , F );
													for( unsigned int d1=0 ; d1<Dim ; d1++ ) for( unsigned int d2=0 ; d2<Dim ; d2++ ) dot += m(d1,d2) * _stencil[i](d1,d2);
												};
											// Iterate over all elements supported on the cell
											Basis< Dim >::ElementSupport( E ).process( __Kernel );
										}
									};
								// Iterate over all cells supported by the pair of functions
								Range< Dim >::Intersect( Basis< Dim >::FunctionSupport( F1 ) , Basis< Dim >::FunctionSupport( F2 ) , eRange ).process( _Kernel );

								if( dot ) S.insert( (int)f2 , (int)f1 ) = dot;
							}
						};
					// Iterate over all functions whose support overlaps f1
					matrixInfo.processAll( F1 , Kernel );
				}
			}
		);
	}
	else
	{
		IntegrationStencil< MishaK::SquareMatrix< double, Dim >, 2, 0 > stencil = ConstantWeightedStiffnessStencil(_r);

		ThreadPool::ParallelFor
		(
			0 , indexer.numFunctions() ,
			[&]( unsigned int t , size_t f1 )
			{
				Index< Dim > F1 = indexer.functionIndex(f1);

				if( supportedFunctions[f1] )
				{
					Window::IsotropicStaticWindow< size_t , Dim , 3 > fNeighbors = indexer.ffNeighbors( f1 , t );
					Window::IsotropicStaticWindow< size_t , Dim , 2 > eNeighbors = indexer.feNeighbors( f1 , t );

					// Precompute the values on the elements supported by F1
					Window::IsotropicStaticWindow< MishaK::SquareMatrix< double , Dim > , Dim , 2 >  ipf;
					{
						auto Kernel = [&]( Hat::Index< Dim > E )
							{
								Hat::Index< Dim > I = E - F1 + Off;
								if( eNeighbors( &I[0] )!=-1 )
									if constexpr( NeedsThreadID ) ipf( &I[0] ) = IPF( E , t );
									else                          ipf( &I[0] ) = IPF( E );
							};
						Range< Dim >::Intersect( Basis< Dim >::FunctionSupport( F1 ) , eRange ).process( Kernel );
					}

					auto Kernel = [&]( Index< Dim > F2 , size_t e , bool flip )
						{
							Hat::Index< Dim > I = F2-F1+Off;
							size_t f2 = fNeighbors( &I[0] );
							if( f2!=-1 && supportedFunctions[f2] )
							{
								double dot = 0;
								auto f = [&]( Index< Dim > E )
									{
										Hat::Index< Dim > I = E - F1 + Off;
										{
											MishaK::SquareMatrix< double, Dim > m = ipf( &I[0] );
											unsigned int i1 = Basis< Dim >::template _RelativeIndex< 0 >( E , F1 );
											unsigned int i2 = Basis< Dim >::template _RelativeIndex< 0 >( E , F2 );
											dot += MishaK::SquareMatrix< double , Dim >::Dot( m , stencil[i1][i2] );
										}
									};
								// Iterate over all cells supported by the i-th function
								Range< Dim >::Intersect( Basis< Dim >::FunctionSupport(F1) , Basis< Dim >::FunctionSupport(F2) , eRange ).process(f);

								if( dot ) S.insert( (int)f2 , (int)f1 ) = dot;
							}
						};
					matrixInfo.processAll( F1 , Kernel );
				}
			}
		);
	}

	S.makeCompressed();
	return S;
}

template< unsigned int Dim >
template< typename ProlongationIndexer /* = Hat::BaseProlongationIndexer< Dim > */ >
Eigen::SparseMatrix< double > ScalarFunctions< Dim >::prolongation( const ProlongationIndexer &pIndexer , size_t numFineFunctions ) const
{
	static_assert( std::is_base_of_v< Hat::BaseProlongationIndexer< Dim > , ProlongationIndexer > , "[ERROR] ProlongationIndexer poorly formed" );

	Eigen::SparseMatrix< double > P( numFineFunctions , pIndexer.numFunctions() );
	Eigen::VectorXi rowSizes( (int)pIndexer.numFunctions() );
	ProlongationStencil pStencil;

	ThreadPool::ParallelFor
	(
		0 , pIndexer.numFunctions() ,
		[&]( unsigned int t , size_t f )
		{
			Index< Dim > F = pIndexer.functionIndex(f);
			Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = pIndexer.ffChildNeighbors( f , t );
			unsigned int sz = 0;
			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 3 >() ; i++ ) if( neighbors.data[i]!=-1 ) sz++;
			rowSizes[f] = sz;
		}
	);

	P.reserve( rowSizes );

	Range< Dim > range;
	Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2*resolution()+1 , Off[d] = 1;

	ThreadPool::ParallelFor
	(
		0 , pIndexer.numFunctions() ,
		[&]( unsigned int t , size_t f1 )
		{
			Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = pIndexer.ffChildNeighbors( f1 , t );
			Index< Dim > F1 = pIndexer.functionIndex(f1)*2;

			auto Kernel = [&]( Index< Dim > F2 )
				{
					Hat::Index< Dim > I = F2 - F1 + Off;
					size_t f2 = neighbors( &I[0] );
					if( f2!=-1 ) P.insert( (int)f2 , (int)f1 ) = pStencil( I );
				};
			Range< Dim >::Intersect( range , Range< Dim >(F1).dilate(1) ).process( Kernel );
		}
	);

	P.makeCompressed();

	return P;
};


template< unsigned int Dim >
template< typename ProlongationIndexer /* = Hat::BaseProlongationIndexer< Dim > */ >
Eigen::VectorXd ScalarFunctions< Dim >::prolongation( const Eigen::VectorXd &x , const ProlongationIndexer &pIndexer , size_t numFineFunctions ) const
{
	static_assert( std::is_base_of_v< Hat::BaseProlongationIndexer< Dim > , ProlongationIndexer > , "[ERROR] ProlongationIndexer poorly formed" );

	Eigen::VectorXd xP = Eigen::VectorXd::Zero( numFineFunctions );

	Range< Dim > range;
	Index< Dim > Off;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2*resolution()+1 , Off[d] = 1;

	ProlongationStencil pStencil;

	ThreadPool::ParallelFor
	(
		0 , pIndexer.numFunctions() ,
		[&]( unsigned int t , size_t f1 )
		{
			Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors = pIndexer.ffChildNeighbors( f1 , t );
			Index< Dim > F1 = pIndexer.functionIndex(f1)*2;

			auto Kernel = [&]( Index< Dim > F2 )
				{
					Hat::Index< Dim > I = F2 - F1 + Off;
					size_t f2 = neighbors( &I[0] );
					if( f2!=-1 ) AddAtomic( xP[f2] , x[f1] * pStencil(I) );
				};
			Range< Dim >::Intersect( range , Range< Dim >(F1).dilate(1) ).process( Kernel );
		}
	);

	return xP;
}
