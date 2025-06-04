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

/////////////////
// BaseIndexer //
/////////////////
template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 3 > BaseIndexer< Dim >::ffNeighbors( size_t f , unsigned int thread ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors;

	Hat::Index< Dim > Off;
	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2 , Off[d] = 1;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors = this->feNeighbors( f , thread );
	auto Kernel = [&]( Hat::Index< Dim > I )
		{
			size_t e = feNeighbors( &I[0] );
			if( e!=-1 )
			{
				Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors = this->efNeighbors( e , thread );
				auto Kernel = [&]( Hat::Index< Dim > _I )
					{
						Hat::Index< Dim > F = I + _I - Off;
						ffNeighbors( &F[0] ) = efNeighbors( &_I[0] );
					};
				range.process( Kernel );
			}
		};
	range.process( Kernel );

	return ffNeighbors;
}

template< unsigned int Dim >
std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > BaseIndexer< Dim >::fNeighbors( size_t f , unsigned int thread ) const
{
	return std::make_pair( ffNeighbors( f , thread ) , feNeighbors( f , thread ) );
}

/////////////////////////////
// BaseHierarchicalIndexer //
/////////////////////////////
template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 3 > BaseHierarchicalIndexer< Dim >::ffNeighbors( unsigned int depth , size_t f , unsigned int thread ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors;

	Hat::Index< Dim > Off;
	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2 , Off[d] = 1;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors = this->feNeighbors( depth , f , thread );
	auto Kernel = [&]( Hat::Index< Dim > I )
		{
			size_t e = feNeighbors( &I[0] );
			if( e!=-1 )
			{
				Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors = this->efNeighbors( depth , e , thread );
				auto Kernel = [&]( Hat::Index< Dim > _I )
					{
						Hat::Index< Dim > F = I + _I - Off;
						ffNeighbors( &F[0] ) = efNeighbors( &_I[0] );
					};
				range.process( Kernel );
			}
		};
	range.process( Kernel );

	return ffNeighbors;
}

template< unsigned int Dim >
std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > BaseHierarchicalIndexer< Dim >::fNeighbors( unsigned int depth , size_t f , unsigned int thread ) const
{
	return std::make_pair( ffNeighbors( depth , f , thread ) , feNeighbors( depth , f , thread ) );
}

template< unsigned int Dim >
Eigen::VectorXd BaseHierarchicalIndexer< Dim >::regularCoefficients( const std::vector< Eigen::VectorXd > &hierarchicalCoefficients ) const
{
	Eigen::VectorXd coefficients;

	Hat::HierarchicalRegularIndexer< Dim > hierarchicalRegularIndexer( maxDepth() );
	auto ToRegular = [&]( unsigned int d , const Eigen::VectorXd &x )
		{
			ScalarFunctions< Dim > scalars( 1<<d );
			Eigen::VectorXd _x = Eigen::VectorXd::Zero( scalars.functionNum() );
			for( size_t f=0 ; f<numFunctions(d) ; f++ ) _x[ scalars.functionIndex( functionIndex( d , f ) ) ] = x[f];
			return _x;
		};

	auto Prolongation = [&]( unsigned int d , const Eigen::VectorXd &x )
		{
			ScalarFunctions< Dim > scalars( 1<<d ) , fineScalars( 1<<(d+1) );
			return scalars.prolongation( x , hierarchicalRegularIndexer[d] , fineScalars.functionNum() );
		};
	coefficients = ToRegular( 0 , hierarchicalCoefficients[0] );
	for( unsigned int d=1 ; d<=maxDepth() ; d++ ) coefficients = ToRegular( d , hierarchicalCoefficients[d] ) + Prolongation( d-1 , coefficients );
	return coefficients;
}

////////////////////////////////
// HierarchicalRegularIndexer //
////////////////////////////////

template< unsigned int Dim >
HierarchicalRegularIndexer< Dim >::HierarchicalRegularIndexer( unsigned int maxDepth )
{
	_scalarFunctions.reserve( maxDepth+1 );
	for( unsigned int d=0 ; d<=maxDepth ; d++ ) _scalarFunctions.emplace_back( 1<<d );
}

template< unsigned int Dim >
HierarchicalRegularIndexer< Dim >::HierarchicalRegularIndexer( BinaryStream &stream )
{
	unsigned int maxDepth;
	stream.read( maxDepth );
	_scalarFunctions.reserve( maxDepth+1 );
	for( unsigned int d=0 ; d<=maxDepth ; d++ ) _scalarFunctions.emplace_back( 1<<d );
}

template< unsigned int Dim >
void HierarchicalRegularIndexer< Dim >::write( BinaryStream &stream , bool ) const
{
	unsigned int maxDepth = (unsigned int)_scalarFunctions.size()-1;
	stream.write( maxDepth );
}

template< unsigned int Dim >
unsigned int HierarchicalRegularIndexer< Dim >::maxDepth( void ) const { return (unsigned int)_scalarFunctions.size()-1; }

template< unsigned int Dim >
size_t HierarchicalRegularIndexer< Dim >::resolution( unsigned int depth ) const { return static_cast< size_t >(1)<<depth; }

template< unsigned int Dim >
size_t HierarchicalRegularIndexer< Dim >::numFunctions( unsigned int depth ) const { return _scalarFunctions[depth].functionNum(); }

template< unsigned int Dim >
size_t HierarchicalRegularIndexer< Dim >::numActiveFunctions( unsigned int depth ) const { return _scalarFunctions[depth].functionNum(); }

template< unsigned int Dim >
size_t HierarchicalRegularIndexer< Dim >::numElements( unsigned int depth ) const { return _scalarFunctions[depth].elementNum(); }

template< unsigned int Dim >
Hat::Index< Dim > HierarchicalRegularIndexer< Dim >::functionIndex( unsigned int depth , size_t f ) const { return _scalarFunctions[depth].functionIndex( f ); }

template< unsigned int Dim >
Hat::Index< Dim > HierarchicalRegularIndexer< Dim >::elementIndex( unsigned int depth , size_t e ) const { return _scalarFunctions[depth].elementIndex( e ); }

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 3 > HierarchicalRegularIndexer< Dim >::ffNeighbors( unsigned int depth , size_t f , unsigned int ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors;
	Hat::Range< Dim > range;
	Hat::Index< Dim > off , I = _scalarFunctions[depth].functionIndex( f );
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = (1<<depth)+1 , off[d] = 1 - I[d];

	for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 3 >() ; i++ ) neighbors.data[i] = static_cast< size_t >(-1);
	auto Kernel = [&]( Hat::Index< Dim > _I )
		{
			Hat::Index< Dim > __I = _I + off;	// __I = _I - I + (1,...,1)
			neighbors( &__I[0] ) = _scalarFunctions[depth].functionIndex(_I);
		};
	Hat::Range< Dim >::Intersect( Hat::Range< Dim >( I ).dilate(1) , range ).process( Kernel );
	return neighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 3 > HierarchicalRegularIndexer< Dim >::ffChildNeighbors( unsigned int depth , size_t f , unsigned int ) const
{
	return ffNeighbors( depth+1 , _scalarFunctions[depth+1].functionIndex( _scalarFunctions[depth].functionIndex(f) * 2 ) );
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalRegularIndexer< Dim >::efNeighbors( unsigned int depth , size_t e , unsigned int ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors;

	Hat::Index< Dim > I = _scalarFunctions[depth].elementIndex(e);
	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2;

	auto Kernel = [&]( Hat::Index< Dim > dI )
		{
			Hat::Index< Dim > _I = I + dI;
			neighbors( &dI[0] ) = _scalarFunctions[depth].functionIndex(_I);
		};
	range.process( Kernel );
	return neighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalRegularIndexer< Dim >::feNeighbors( unsigned int depth , size_t f , unsigned int ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors;

	Hat::Index< Dim > F = _scalarFunctions[depth].functionIndex(f) , Off;
	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2 , Off[d] = -1;

	unsigned int res = 1<<depth;
	auto Interior = [res]( Index< Dim > E )
		{
			for( unsigned int d=0 ; d<Dim ; d++ ) if( E[d]<0 || E[d]>=(int)res ) return false;
			return true;
		};

	auto Kernel = [&]( Hat::Index< Dim > dE )
		{
			Hat::Index< Dim > E = F + dE + Off;
			if( Interior( E ) ) neighbors( &dE[0] ) = _scalarFunctions[depth].elementIndex(E);
			else                neighbors( &dE[0] ) = -1;
		};
	range.process( Kernel );
	return neighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalRegularIndexer< Dim >::fNeighbors( unsigned int depth , Point< double , Dim > p , unsigned int ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors;

	unsigned int res = 1<<depth;

#ifdef DEBUG_HAT_ADAPTIVE
	for( unsigned int d=0 ; d<Dim ; d++ ) if( p[d]<0 || p[d]>1 ) MK_ERROR_OUT( "Point out of bounds: " , p );
#endif // DEBUG_HAT_ADAPTIVE
	Hat::Index< Dim > I;
	for( unsigned int d=0 ; d<Dim ; d++ ) I[d] = std::max< int >( 0 , std::min< int >( (int)floor( p[d] * res ) , res-1 ) );
	Hat::Range< Dim > range;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2;

	auto F = [&]( Hat::Index< Dim > dI )
		{
			Hat::Index< Dim > _I = I + dI;
			neighbors( &dI[0] ) = _scalarFunctions[depth].functionIndex(_I);
		};
	range.process( F );
	return neighbors;
}


template< unsigned int Dim >
size_t HierarchicalRegularIndexer< Dim >::parentElement( unsigned int depth , size_t e , unsigned int ) const
{
	if( depth )
	{
		Hat::Index< Dim > E = _scalarFunctions[depth].elementIndex(e);
		for( unsigned int d=0 ; d<Dim ; d++ ) E[d] >>= 1;
		return _scalarFunctions[depth-1].elementIndex( E );
	}
	else return -1;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalRegularIndexer< Dim >::childElements( unsigned int depth , size_t e , unsigned int ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > cElements;
	if( depth<_scalarFunctions.size()-1 )
	{
		Hat::Range< Dim > range;
		for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2;
		Hat::Index< Dim > E = _scalarFunctions[depth].elementIndex(e) * 2;
		auto Kernel = [&]( Hat::Index< Dim > dE ){ cElements( &dE[0] ) = _scalarFunctions[depth+1].elementIndex( E + dE ); };
		range.process( Kernel );
	}
	else for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ ) cElements.data[i] = -1;
	return cElements;
}

////////////////////////////////
// HierarchicalAdaptedIndexer //
////////////////////////////////

// [NOTE]  "depth" will refer to external depth
//        "_depth" will refer to internal depth

template< unsigned int Dim >
template< typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ >
HierarchicalAdaptedIndexer< Dim >::HierarchicalAdaptedIndexer( size_t sampleNum , SampleFunctor && Samples , unsigned int kernelRadius , unsigned int maxDepth )
{
	_root.template initChildren< false >( nullptr );
	_spaceRoot = _root.children;

	unsigned int _maxDepth = maxDepth+1;
	_ffKeys.resize( ThreadPool::NumThreads() );
	_efKeys.resize( ThreadPool::NumThreads() );
	_feKeys.resize( ThreadPool::NumThreads() );
	for( unsigned int i=0 ; i<ThreadPool::NumThreads() ; i++ ) _ffKeys[i].set( _maxDepth ) , _efKeys[i].set( _maxDepth ) , _feKeys[i].set( _maxDepth );

	switch( kernelRadius )
	{
	case  0: _addSamples<  0 >( sampleNum , std::forward< SampleFunctor >( Samples ) , maxDepth ) ; break;
	case  1: _addSamples<  1 >( sampleNum , std::forward< SampleFunctor >( Samples ) , maxDepth ) ; break;
	case  2: _addSamples<  2 >( sampleNum , std::forward< SampleFunctor >( Samples ) , maxDepth ) ; break;
	case  3: _addSamples<  3 >( sampleNum , std::forward< SampleFunctor >( Samples ) , maxDepth ) ; break;
	case  4: _addSamples<  4 >( sampleNum , std::forward< SampleFunctor >( Samples ) , maxDepth ) ; break;
	case  5: _addSamples<  5 >( sampleNum , std::forward< SampleFunctor >( Samples ) , maxDepth ) ; break;
	default: MK_ERROR_OUT( "Only kernel radius in the range [0,5] supported: " , kernelRadius );
	}

	_finalize( maxDepth );
}

template< unsigned int Dim >
HierarchicalAdaptedIndexer< Dim >::HierarchicalAdaptedIndexer( BinaryStream &stream )
{
	_root.read( stream , nullptr );
	stream.read( _activeSizes );
	stream.read( _functionNodes );
	stream.read( _elementNodes );

	_spaceRoot = _root.children;
	unsigned int _maxDepth = _root.maxDepth();
	unsigned int maxDepth = _maxDepth-1;
	_ffKeys.resize( ThreadPool::NumThreads() );
	_efKeys.resize( ThreadPool::NumThreads() );
	_feKeys.resize( ThreadPool::NumThreads() );
	for( unsigned int i=0 ; i<ThreadPool::NumThreads() ; i++ ) _ffKeys[i].set( _maxDepth ) , _efKeys[i].set( _maxDepth ) , _feKeys[i].set( _maxDepth );

	auto Kernel = [&]( Node * node )
		{
			int _depth = node->depth();
			if( _depth>0 )
			{
				unsigned int depth = _depth-1;
				if( node->nodeData.functionIndex!=-1 ) _functionNodes[depth][node->nodeData.functionIndex] = node;
				if( node->nodeData.elementIndex!=-1 ) _elementNodes[depth][node->nodeData.elementIndex] = node;
			}
		};
	_root.processNodes( Kernel );
}

template< unsigned int Dim >
void HierarchicalAdaptedIndexer< Dim >::write( BinaryStream &stream , bool serialize ) const
{
	_root.write( stream , serialize );
	stream.write( _activeSizes );
	stream.write( _functionNodes );
	stream.write( _elementNodes );
}

template< unsigned int Dim >
unsigned int HierarchicalAdaptedIndexer< Dim >::maxDepth( void ) const { return (unsigned int)_functionNodes.size()-1; }

template< unsigned int Dim >
size_t HierarchicalAdaptedIndexer< Dim >::resolution( unsigned int depth ) const { return static_cast< size_t >(1)<<depth; }

template< unsigned int Dim >
size_t HierarchicalAdaptedIndexer< Dim >::numFunctions( unsigned int depth ) const { return _functionNodes[depth].size(); }

template< unsigned int Dim >
size_t HierarchicalAdaptedIndexer< Dim >::numActiveFunctions( unsigned int depth ) const { return _activeSizes[depth]; }

template< unsigned int Dim >
size_t HierarchicalAdaptedIndexer< Dim >::numElements( unsigned int depth ) const { return _elementNodes[depth].size(); }

template< unsigned int Dim >
typename HierarchicalAdaptedIndexer< Dim >::NodeData & HierarchicalAdaptedIndexer< Dim >::_functionNodeData( unsigned int depth , size_t f ){ return _functionNodes[depth][f]->nodeData; }

template< unsigned int Dim >
const typename HierarchicalAdaptedIndexer< Dim >::NodeData & HierarchicalAdaptedIndexer< Dim >::_functionNodeData( unsigned int depth , size_t f ) const { return _functionNodes[depth][f]->nodeData; }

template< unsigned int Dim >
typename HierarchicalAdaptedIndexer< Dim >::NodeData & HierarchicalAdaptedIndexer< Dim >::_elementNodeData( unsigned int depth , size_t f ){ return _elementNodes[depth][f]->nodeData; }

template< unsigned int Dim >
const typename HierarchicalAdaptedIndexer< Dim >::NodeData & HierarchicalAdaptedIndexer< Dim >::_elementNodeData( unsigned int depth , size_t f ) const { return _elementNodes[depth][f]->nodeData; }

template< unsigned int Dim >
Hat::Index< Dim > HierarchicalAdaptedIndexer< Dim >::functionIndex( unsigned int depth , size_t f ) const { return _functionNodes[depth][f]->offset(); }

template< unsigned int Dim >
Hat::Index< Dim > HierarchicalAdaptedIndexer< Dim >::elementIndex( unsigned int depth , size_t e ) const { return _elementNodes[depth][e]->offset(); }

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 3 > HierarchicalAdaptedIndexer< Dim >::ffNeighbors( unsigned int depth , size_t f , unsigned int thread ) const
{
	unsigned int _depth = depth+1;

	Window::IsotropicStaticWindow< size_t , Dim , 3 > neighbors;

	ConstFunctionFunctionNeighborKey &ffKey = const_cast< ConstFunctionFunctionNeighborKey & >( _ffKeys[thread] );

	ffKey.getNeighbors( _functionNodes[depth][f] );
	typename ConstFunctionFunctionNeighborKey::NeighborType &_neighbors = ffKey.neighbors[_depth];

	for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 3 >() ; i++ )
		if( _neighbors.neighbors.data[i] ) neighbors.data[i] = _neighbors.neighbors.data[i]->nodeData.functionIndex;
		else                               neighbors.data[i] = -1;

	return neighbors;
}

template< unsigned int Dim >
std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > HierarchicalAdaptedIndexer< Dim >::fNeighbors( unsigned int depth , size_t f , unsigned int thread ) const
{
	static bool firstTime = true;
	static unsigned int f2e[ Window::template IsotropicSize< Dim , 3 >() ];
	if( firstTime )
	{
		firstTime = false;
		Hat::Range< Dim > range;
		for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 3;

		auto Kernel = [&]( Hat::Index< Dim > I )
			{
				bool inRange = true;
				for( unsigned int d=0 ; d<Dim ; d++ ) inRange &= I[d]<2;
				if( inRange ) f2e[ Window::IsotropicGetIndex< Dim , 3 >( &I[0] ) ] = Window::IsotropicGetIndex< Dim , 2 >( &I[0] );
				else          f2e[ Window::IsotropicGetIndex< Dim , 3 >( &I[0] ) ] = -1;
			};
		range.process( Kernel );
	}

	unsigned int _depth = depth+1;

	std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > neighbors;

	ConstFunctionFunctionNeighborKey &ffKey = const_cast< ConstFunctionFunctionNeighborKey & >( _ffKeys[thread] );

	ffKey.getNeighbors( _functionNodes[depth][f] );
	typename ConstFunctionFunctionNeighborKey::NeighborType &_neighbors = ffKey.neighbors[_depth];

	for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 3 >() ; i++ )
		if( _neighbors.neighbors.data[i] )
		{
			neighbors.first.data[i] = _neighbors.neighbors.data[i]->nodeData.functionIndex;
			if( f2e[i]!=-1 ) neighbors.second.data[ f2e[i] ] = _neighbors.neighbors.data[i]->nodeData.elementIndex;
		}
		else
		{
			neighbors.first.data[i] = -1;
			if( f2e[i]!=-1 ) neighbors.second.data[ f2e[i] ] = -1;
		}

	return neighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 3 > HierarchicalAdaptedIndexer< Dim >::ffChildNeighbors( unsigned int depth , size_t f , unsigned int thread ) const
{
	unsigned int _depth = depth+1;

	Window::IsotropicStaticWindow< size_t , Dim , 3 > childNeighbors;

	ConstFunctionFunctionNeighborKey &ffKey = const_cast< ConstFunctionFunctionNeighborKey & >( _ffKeys[thread] );
	ffKey.getNeighbors( _functionNodes[depth][f] );
	typename ConstFunctionFunctionNeighborKey::NeighborType _childNeighbors;

	ffKey.getChildNeighbors( 0 , _depth , _childNeighbors );

	for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 3 >() ; i++ )
		if( _childNeighbors.neighbors.data[i] ) childNeighbors.data[i] = _childNeighbors.neighbors.data[i]->nodeData.functionIndex;
		else                                    childNeighbors.data[i] = -1;

	return childNeighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalAdaptedIndexer< Dim >::efNeighbors( unsigned int depth , size_t e , unsigned int thread ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors;
	ConstElementFunctionNeighborKey &efKey = const_cast< ConstElementFunctionNeighborKey & >( _efKeys[thread] );

	unsigned int _depth = depth+1;
	efKey.getNeighbors( _elementNodes[depth][e] );
	typename ConstElementFunctionNeighborKey::NeighborType &_neighbors = efKey.neighbors[_depth];

	for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 2 >() ; i++ )
		if( _neighbors.neighbors.data[i] ) neighbors.data[i] = _neighbors.neighbors.data[i]->nodeData.functionIndex;
		else                               neighbors.data[i] = -1;
	return neighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalAdaptedIndexer< Dim >::feNeighbors( unsigned int depth , size_t f , unsigned int thread ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors;
	ConstFunctionElementNeighborKey &feKey = const_cast< ConstFunctionElementNeighborKey & >( _feKeys[thread] );

	unsigned int _depth = depth+1;
	feKey.getNeighbors( _functionNodes[depth][f] );
	typename ConstFunctionElementNeighborKey::NeighborType &_neighbors = feKey.neighbors[_depth];

	for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 2 >() ; i++ )
		if( _neighbors.neighbors.data[i] ) neighbors.data[i] = _neighbors.neighbors.data[i]->nodeData.elementIndex;
		else                               neighbors.data[i] = -1;
	return neighbors;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalAdaptedIndexer< Dim >::fNeighbors( unsigned int depth , Point< double , Dim > p , unsigned int thread ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > neighbors;
	ConstElementFunctionNeighborKey &efKey = const_cast< ConstElementFunctionNeighborKey & >( _efKeys[thread] );

	unsigned int _depth = depth+1;
	const Node *node = _spaceRoot->getNode( p , depth );
	if( !node )
	{
		MK_WARN( "Could not find point @ depth: " , p , " @ " , depth );
		for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 2 >() ; i++ ) neighbors.data[i] = -1;
	}
	else
	{
		efKey.getNeighbors( node );
		typename ConstElementFunctionNeighborKey::NeighborType &_neighbors = efKey.neighbors[_depth];

		for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 2 >() ; i++ )
			if( _neighbors.neighbors.data[i] ) neighbors.data[i] = _neighbors.neighbors.data[i]->nodeData.functionIndex;
			else                               neighbors.data[i] = -1;
	}
	return neighbors;
}

template< unsigned int Dim >
size_t HierarchicalAdaptedIndexer< Dim >::parentElement( unsigned int depth , size_t e , unsigned int ) const
{
	if( depth ) return _elementNodes[depth][e]->nodeData.elementIndex;
	else return -1;
}

template< unsigned int Dim >
Window::IsotropicStaticWindow< size_t , Dim , 2 > HierarchicalAdaptedIndexer< Dim >::childElements( unsigned int depth , size_t e , unsigned int ) const
{
	Window::IsotropicStaticWindow< size_t , Dim , 2 > cElements;
	const Node *node = _elementNodes[depth][e];
	if( node->children )
		for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
		{
			Hat::Index< Dim > I = Hat::ScalarFunctions< Dim >::ElementIndex( i , 2 );
			cElements( &I[0] ) = node->children[i].nodeData.elementIndex;
		}
	else for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ ) cElements.data[i] = -1;
	return cElements;
}

template< unsigned int Dim >
template< unsigned int KernelRadius , typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ >
void HierarchicalAdaptedIndexer< Dim >::_addSamples( size_t sampleNum , SampleFunctor && Samples , unsigned int maxDepth )
{
	auto InteriorFunction = [&]( const Node *node )
		{
			int depth;
			Hat::Index< Dim > offset;
			node->depthAndOffset( depth , offset );
			depth--;
			bool interior = true;
			for( unsigned int d=0 ; d<Dim ; d++ ) interior &= offset[d]>=0 && offset[d]<=(1<<depth);
			return interior;
		};

	unsigned int _maxDepth = maxDepth+1;
	ElementFunctionNeighborKey< KernelRadius > efKey;
	efKey.set( _maxDepth );

	unsigned int res = 1<<maxDepth;
	OrderedSamples< Dim > samples( std::forward< SampleFunctor >( Samples ) , sampleNum , 1<<maxDepth );
	for( size_t i=0 ; i<samples.size() ; i++ )
	{
		Hat::Index< Dim > E = samples[i].first;
		const std::vector< Point< double , Dim > > &subSamples = samples[i].second;

		Point< double , Dim > p;
		for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d] + 0.5 ) / res;
		Node *n = _spaceRoot->template insertPoint< false >( p , maxDepth , nullptr );
		efKey.template getNeighbors< true , false >( n , nullptr );

		for( unsigned int _d=_maxDepth ; _d>=1 ; _d-- )
		{
			typename ElementFunctionNeighborKey< KernelRadius >::NeighborType &neighbors = efKey.neighbors[_d];
			for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 2*KernelRadius+2 >() ; i++ ) if( neighbors.neighbors.data[i] )
				if( InteriorFunction( neighbors.neighbors.data[i] ) )
					neighbors.neighbors.data[i]->nodeData.functionIndex = 0;
		}
	}

	_activeSizes.resize( maxDepth+1 , 0 );
	{
		auto NodeFunctor = [&]( Node * node )
			{
				int _depth = node->depth();
				if( _depth>0 )
				{
					unsigned int depth = _depth-1;
					if( node->nodeData.functionIndex!=-1 ) node->nodeData.functionIndex = _activeSizes[depth]++;
				}
			};
		_root.processNodes( NodeFunctor );
	}

	_functionNodes.resize( maxDepth+1 );
	for( unsigned int d=0 ; d<=maxDepth ; d++ ) _functionNodes[d].resize( _activeSizes[d] );

	{
		auto NodeFunctor = [&]( Node * node )
			{
				int _depth = node->depth();
				if( _depth>0 )
				{
					unsigned int depth = _depth-1;
					if( node->nodeData.functionIndex!=-1 ) _functionNodes[depth][node->nodeData.functionIndex] = node;
				}
			};
		_root.processNodes( NodeFunctor );
	}
}


template< unsigned int Dim >
void HierarchicalAdaptedIndexer< Dim >::_finalize( unsigned int maxDepth )
{
	auto InteriorFunction = [&]( const Node *node )
		{
			int depth;
			Hat::Index< Dim > offset;
			node->depthAndOffset( depth , offset );
			depth--;
			bool interior = true;
			for( unsigned int d=0 ; d<Dim ; d++ ) interior &= offset[d]>=0 && offset[d]<=(1<<depth);
			return interior;
		};

	auto InteriorElement = [&]( const Node *node )
		{
			int depth;
			Hat::Index< Dim > offset;
			node->depthAndOffset( depth , offset );
			depth--;
			bool interior = true;
			for( unsigned int d=0 ; d<Dim ; d++ ) interior &= offset[d]>=0 && offset[d]<(1<<depth);
			return interior;
		};

	unsigned int _maxDepth = maxDepth+1;

	// A key for funding the functions incident on a function (and also for funding elements incident on a function)
	using NeighborKey = typename Node::template NeighborKey< ParameterPack::IsotropicUIntPack< Dim , 1 > , ParameterPack::IsotropicUIntPack< Dim , 1 > >;
	NeighborKey nKey;
	nKey.set( maxDepth+1 );

	std::vector< size_t > fSizes = _activeSizes;
	std::vector< size_t > eSizes( maxDepth+1 , 0 );

	// Currently all indexed functions are active
	for( unsigned int _d=_maxDepth ; _d>=1 ; _d-- )
	{
		unsigned int d = _d-1;
		for( unsigned int i=0 ; i<_functionNodes[d].size() ; i++ )
		{
			Node *node = _functionNodes[d][i];
			nKey.template getNeighbors< true , false >( node , nullptr );

			Hat::Index< Dim > I = node->offset();
			typename NeighborKey::NeighborType &neighbors = nKey.neighbors[_d];

			for( unsigned int i=0 ; i<Window::template IsotropicSize< Dim , 3 >() ; i++ )
			{
				Node *_n = neighbors.neighbors.data[i];
				if( _n )
				{
					if( _n->nodeData.functionIndex==-1 && InteriorFunction(_n) ) _n->nodeData.functionIndex = fSizes[d]++;
					if( _n->nodeData.elementIndex==-1 && InteriorElement(_n) )
					{
						Hat::Index< Dim > _I = _n->offset();
						bool elementNbr = true;
						for( unsigned int d=0 ; d<Dim ; d++ ) elementNbr &= _I[d]<=I[d];
						if( elementNbr ) _n->nodeData.elementIndex = eSizes[d]++;
					}
				}
			}
			node = node->parent;
		}
	}

	_elementNodes.resize( maxDepth+1 );
	for( unsigned int d=0 ; d<=maxDepth ; d++ ) _functionNodes[d].resize( fSizes[d] ) , _elementNodes[d].resize( eSizes[d] );
	{
		auto NodeFunctor = [&]( Node * node )
			{
				int _depth = node->depth();
				if( _depth>0 )
				{
					unsigned int depth = _depth-1;
					if( node->nodeData.functionIndex!=-1 ) _functionNodes[depth][node->nodeData.functionIndex] = node;
					if( node->nodeData.elementIndex!=-1 ) _elementNodes[depth][node->nodeData.elementIndex] = node;
				}
			};
		_root.processNodes( NodeFunctor );
	}
}

template< unsigned int Dim >
const typename HierarchicalAdaptedIndexer< Dim >::Node & HierarchicalAdaptedIndexer< Dim >::root( void ) const { return _root; }
