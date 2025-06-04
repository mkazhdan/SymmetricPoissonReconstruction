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

template< unsigned int Dim , unsigned int CoDim >
IsoTree< Dim , CoDim >::IsoTree( unsigned int maxDepth )
{
	_root.template initChildren< false >( nullptr );
	_spaceRoot = _root.children;
	_maxDepth = maxDepth+1;
	_efKeys.resize( ThreadPool::NumThreads() );
	_efKey.set( _maxDepth );
	_eeKey.set( _maxDepth );
	for( unsigned int i=0 ; i<ThreadPool::NumThreads() ; i++ ) _efKeys[i].set( _maxDepth );
}

template< unsigned int Dim , unsigned int CoDim >
IsoTree< Dim , CoDim >::IsoTree( void ) : IsoTree( 0 ){}

template< unsigned int Dim , unsigned int CoDim >
template< typename HierarchicalIndexer /* = Hat::BaseHierarchicalIndexer< Dim > */ , typename CoefficientFunctor /* = std::function< Point< double , CoDim > (unsigned int , size_t ) > */ >
IsoTree< Dim , CoDim >::IsoTree( const HierarchicalIndexer & hierarchicalIndexer , CoefficientFunctor && F )
	: IsoTree( hierarchicalIndexer , std::forward< CoefficientFunctor >( F ) , hierarchicalIndexer.maxDepth() )
{}

template< unsigned int Dim , unsigned int CoDim >
template< typename HierarchicalIndexer /* = Hat::BaseHierarchicalIndexer< Dim > */ , typename CoefficientFunctor /* = std::function< Point< double , CoDim > (unsigned int , size_t ) > */ >
IsoTree< Dim , CoDim >::IsoTree( const HierarchicalIndexer & hierarchicalIndexer , CoefficientFunctor && F , unsigned int maxDepth )
	: IsoTree( std::min< unsigned int >( maxDepth , hierarchicalIndexer.maxDepth() ) )
{
	static_assert( std::is_base_of_v< Hat::BaseHierarchicalIndexer< Dim > , HierarchicalIndexer > , "[ERROR] HierarchicalIndexer poorly formed" );
	static_assert( std::is_convertible_v< CoefficientFunctor , std::function< Point< double , CoDim > (unsigned int , size_t ) > > , "[ERROR] CoefficientFunctor poorly formed" );

	if( maxDepth>hierarchicalIndexer.maxDepth() )
	{
		MK_WARN( "Max depth exceeds indexer depth: " , maxDepth , " -> " , hierarchicalIndexer.maxDepth() );
		maxDepth = hierarchicalIndexer.maxDepth();
	}

	// Set the level-local values
	for( unsigned int d=0 ; d<=maxDepth ; d++ )
	{
		const auto & indexer = hierarchicalIndexer[d];
		for( size_t f=0 ; f<indexer.numFunctions() ; f++ )
		{
			Node *node = _addFunction( indexer.functionIndex(f) , d );
			node->nodeData.v = F(d,f);
		}
	}

	for( unsigned int d=0 ; d<=maxDepth ; d++ )
	{
		const auto & indexer = hierarchicalIndexer[d];
		for( size_t e=0 ; e<indexer.numElements() ; e++ ) addElement( indexer.elementIndex(e) , d );
	}
}

template< unsigned int Dim , unsigned int CoDim >
typename IsoTree< Dim , CoDim >::Node & IsoTree< Dim , CoDim >::spaceRoot( void ){ return *_spaceRoot; }

template< unsigned int Dim , unsigned int CoDim >
const typename IsoTree< Dim , CoDim >::Node & IsoTree< Dim , CoDim >::spaceRoot( void ) const { return *_spaceRoot; }

template< unsigned int Dim , unsigned int CoDim >
typename IsoTree< Dim , CoDim >::Node * IsoTree< Dim , CoDim >::addElement( Hat::Index< Dim > E , unsigned int depth )
{
	Node *node = _addElement( E , depth );
	if( !node->nodeData.validElement ) _setElementCumulative( node );
	return node;
}

template< unsigned int Dim , unsigned int CoDim >
Point< double , CoDim > IsoTree< Dim , CoDim >::value( Point< double , Dim > p , unsigned int thread , bool forceMaxDepth ) const
{
	Point< double , CoDim > value;

	const Node *eNode = _spaceRoot->getLeafNode( p );
	if( !eNode->nodeData.validElement ) MK_ERROR_OUT( "Leaf node is not a valid element node: " , eNode->nodeData.validElement , " : " , eNode->depth() , " @ " , eNode->offset() );
	if( forceMaxDepth && eNode->depth()!=_maxDepth ) MK_ERROR_OUT( "Leaf node not at max depth: " , eNode->depth() , " / " , _maxDepth , " : " , eNode->offset() );

	unsigned int res = 1<<( eNode->depth()-1 );

	Hat::Range< Dim > range;
	Hat::Index< Dim > E;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2 , E[d] = std::max< int >( 0 , std::min< int >( (int)floor( p[d]*res ) , res-1 ) );

	Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > eValues = elementValues( eNode , thread );

	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = p[d] * res - E[d];

	auto Kernel = [&]( Hat::Index< Dim > I )
		{
			double weight = 1.;
			for( unsigned int d=0 ; d<Dim ; d++ ) weight *= ( I[d]==1 ) ? p[d] : 1.-p[d];
			value += eValues( &I[0] ) * weight;
		};
	range.process( Kernel );

	return value;
}

template< unsigned int Dim , unsigned int CoDim >
Point< Point< double , Dim > , CoDim , double > IsoTree< Dim , CoDim >::gradient( Point< double , Dim > p , unsigned int thread , bool forceMaxDepth ) const
{
	Point< Point< double , Dim > , CoDim , double > grad;

	const Node *eNode = _spaceRoot->getLeafNode( p );
	if( !eNode->nodeData.validElement ) MK_ERROR_OUT( "Leaf node is not a valid element node: " , eNode->nodeData.validElement , " : " , eNode->depth() , " @ " , eNode->offset() );
	if( forceMaxDepth && eNode->depth()!=_maxDepth ) MK_ERROR_OUT( "Leaf node not at max depth: " , eNode->depth() , " / " , _maxDepth , " : " , eNode->offset() );

	unsigned int res = 1<<( eNode->depth()-1 );

	Hat::Range< Dim > range;
	Hat::Index< Dim > E;
	for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = 2 , E[d] = std::max< int >( 0 , std::min< int >( (int)floor( p[d]*res ) , res-1 ) );

	Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > eValues = elementValues( eNode , thread );

	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = p[d] * res - E[d];

	auto Kernel = [&]( Hat::Index< Dim > I )
		{
			Point< double , CoDim > value = eValues( &I[0] );
			for( unsigned int d=0 ; d<Dim ; d++ )
			{
				double weight = res;
				for( unsigned int dd=0 ; dd<Dim ; dd++ )
				{
					if( dd==d ) weight *= ( I[dd]==1 ) ? 1. : -1.;
					else        weight *= ( I[dd]==1 ) ? p[dd] : 1.-p[dd];
				}
				for( unsigned int c=0 ; c<CoDim ; c++ ) grad[c][d] += value[c] * weight;
			}
		};
	range.process( Kernel );

	return grad;
}

template< unsigned int Dim , unsigned int CoDim >
bool IsoTree< Dim , CoDim >::isZeroCrossingNode( const Node * node , bool all , unsigned int thread ) const
{
	struct ZeroCrossingFlags
	{
		ZeroCrossingFlags( void ){ for( unsigned int c=0 ; c<CoDim ; c++ ) flags[c] = 0; }
		unsigned char flags[CoDim];
		unsigned char & operator[]( unsigned int c ){ return flags[c]; }
	};

	ZeroCrossingFlags flags;
	Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > values = elementValues( node , thread );
	for( unsigned int i=0 ; i<Window::IsotropicSize< Dim,  2 >() ; i++ ) for( unsigned int c=0 ; c<CoDim ; c++ )
		if     ( values.data[i][c]<0 ) flags[c] |= 1;
		else if( values.data[i][c]>0 ) flags[c] |= 2;
		else                           flags[c] |= 3; // Shouldn't happen

	if( all )
	{
		bool isZeroCrossing = true;
		for( unsigned int c=0 ; c<CoDim ; c++ ) isZeroCrossing &= ( flags[c]==3 );
		return isZeroCrossing;
	}
	else
	{
		bool isZeroCrossing = false;
		for( unsigned int c=0 ; c<CoDim ; c++ ) isZeroCrossing |= ( flags[c]==3 );
		return isZeroCrossing;
	}
}

template< unsigned int Dim , unsigned int CoDim >
void IsoTree< Dim , CoDim >::refineZeroCrossing( bool all )
{
	static const unsigned char FINALIZED = 1;
	static const unsigned char ADDED = 2;

	std::vector< Node * > activeNodes;
	{
		auto Kernel = [&]( Node *node )
			{
				if( node->depth()==_maxDepth && node->nodeData.validElement )
					if( isZeroCrossingNode( node , all , 0 ) ) activeNodes.push_back( node ) , node->nodeData.state |= ADDED;
					else node->nodeData.state |= FINALIZED;
			};
		spaceRoot().processNodes( Kernel );
	}

	while( activeNodes.size() )
	{
		Node * node = activeNodes.back() ; activeNodes.pop_back();
		if( !( node->nodeData.state & FINALIZED ) )
		{
			Hat::Index< Dim > E = node->offset();
			auto FaceAdjacent = [&]( Node * _node )
				{
					Hat::Index< Dim > _E = _node->offset();
					unsigned int count = 0;
					for( unsigned int d=0 ; d<Dim ; d++ ) if( _E[d]!=E[d] ) count++;
					return count==1;
				};
			Window::IsotropicStaticWindow< typename IsoTree< Dim , CoDim >::Node * , Dim , 3 > neighbors = addElementNeighbors( node , FaceAdjacent );

			for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 3 >() ; i++ )
				if( neighbors.data[i] && !( neighbors.data[i]->nodeData.state & ADDED ) && !( neighbors.data[i]->nodeData.state & FINALIZED ) )
					if( isZeroCrossingNode( neighbors.data[i] , all , 0 ) ) activeNodes.push_back( neighbors.data[i] ) , neighbors.data[i]->nodeData.state |= ADDED;
					else neighbors.data[i]->nodeData.state |= FINALIZED;

			node->nodeData.state |= FINALIZED;
		}
	}
}

template< unsigned int Dim , unsigned int CoDim >
std::vector< const typename IsoTree< Dim , CoDim >::Node * > IsoTree< Dim , CoDim >::levelSetNodes( bool all ) const
{
	std::vector< const typename IsoTree< Dim , CoDim >::Node * > levelSetNodes;
	{
		auto Kernel = [&]( const Node * node ){ if( node->depth()==_maxDepth && node->nodeData.validElement && isZeroCrossingNode( node , all , 0 ) ) levelSetNodes.push_back( node ); };
		spaceRoot().processNodes( Kernel );
	}

	return levelSetNodes;
}

template< unsigned int Dim , unsigned int CoDim >
template< typename AddFunctor /* = std::function< bool ( Node * ) > */ >
Window::IsotropicStaticWindow< typename IsoTree< Dim , CoDim >::Node * , Dim , 3 > IsoTree< Dim , CoDim >::addElementNeighbors( Node * node , AddFunctor && F )
{
	static_assert( std::is_convertible_v< AddFunctor , std::function< bool ( Node * node ) > > , "[ERROR] AddFunctor poorly formed" );
	Window::IsotropicStaticWindow< Node * , Dim , 3 > neighbors;
	_eeKey.template getNeighbors< true , false >( node , nullptr );
	typename ElementElementNeighborKey::NeighborType &_neighbors = _eeKey.neighbors[ node->depth() ];

	unsigned int res = 1<<( node->depth()-1 );
	auto InteriorElementNode = [&]( Node *node )
		{
			Hat::Index< Dim > E = node->offset();
			bool interior = true;
			for( unsigned int d=0 ; d<Dim ; d++ ) if( E[d]<0 || E[d]>=(int)res ) interior = false;
			return interior;
		};

	for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 3 >() ; i++ )
	{
		Node *_node = _neighbors.neighbors.data[i];
		if( _node && InteriorElementNode( _node ) && F( _node ) )
		{
			_setElementCumulative( _node );
			neighbors.data[i] = _node;
		}
		else neighbors.data[i] = nullptr;
	}

	return neighbors;
}


template< unsigned int Dim , unsigned int CoDim >
Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > IsoTree< Dim , CoDim >::elementValues( const Node * node , unsigned int thread ) const
{
	if( !node || !node->nodeData.validElement ) MK_THROW( "Expected valid element node" );

	Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > eValues;
	ConstElementFunctionNeighborKey &efKey = const_cast< ConstElementFunctionNeighborKey & >( _efKeys[thread] );
	efKey.getNeighbors( node );
	typename ConstElementFunctionNeighborKey::NeighborType &neighbors = efKey.neighbors[ node->depth() ];
	for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
	{
#ifdef DEBUG_ISO_TREE
		if( !neighbors.neighbors.data[i] ) MK_ERROR_OUT( "Expected valid function node" );
#endif // DEBUG_ISO_TREE
		eValues.data[i] = neighbors.neighbors.data[i]->nodeData.v;
	}
	return eValues;
}

template< unsigned int Dim , unsigned int CoDim >
Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > IsoTree< Dim , CoDim >::elementValues( Hat::Index< Dim > E , unsigned int thread ) const
{
	return elementValues( _getNode( E , _maxDepth-1 ) , thread );
}

template< unsigned int Dim , unsigned int CoDim >
Point< double , CoDim > IsoTree< Dim , CoDim >::functionValue( Hat::Index< Dim > F ) const
{
	return _getNode( F , _maxDepth-1 )->nodeData.v;
}

template< unsigned int Dim , unsigned int CoDim >
typename IsoTree< Dim , CoDim >::Node * IsoTree< Dim , CoDim >::_addFunction( Hat::Index< Dim > F , unsigned int depth )
{
	unsigned int res = 1<<(depth+1);
	Point< double , Dim > p;
	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( F[d] + 0.5 ) / res;
	return _root.template insertPoint< false >( p , depth+1 , nullptr );
}

template< unsigned int Dim , unsigned int CoDim >
typename IsoTree< Dim , CoDim >::Node * IsoTree< Dim , CoDim >::_addElement( Hat::Index< Dim > E , unsigned int depth )
{
	unsigned int res = 1<<depth;
	Point< double , Dim > p;
	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( E[d] + 0.5 ) / res;
	return _spaceRoot->template insertPoint< false >( p , depth , nullptr );
}

template< unsigned int Dim , unsigned int CoDim >
void IsoTree< Dim , CoDim >::_setElementCumulative( Node * node )
{
	_efKey.template getNeighbors< true , false >( node , nullptr );

	std::function< void ( Node * node ) > SetFunctionCumulative = [&]( Node * node )
		{
			unsigned int _depth = node->depth();
			unsigned int depth = _depth-1;

			if( !_depth );
			else if( !depth );
			else
			{
				Hat::Index< Dim > I = node->offset();
				typename ElementFunctionNeighborKey::NeighborType &pNeighbors = _efKey.neighbors[_depth-1];

				Point< double , CoDim > cumV;
				for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
				{
#ifdef DEBUG_ISO_TREE
					if( !pNeighbors.neighbors.data[i] ) MK_ERROR_OUT( "Expected valid parent function node: " , depth , " @ " , I );
#endif // DEBUG_ISO_TREE
					Hat::Index< Dim > _I = pNeighbors.neighbors.data[i]->offset() * 2;
					double weight = 1.;
					for( unsigned int d=0 ; d<Dim ; d++ )
						if     ( _I[d]==I[d] )                    weight *= 1.0;
						else if( _I[d]==I[d]+1 || _I[d]==I[d]-1 ) weight *= 0.5;
						else                                      weight *= 0.0;
					if( weight )
					{
						if( !pNeighbors.neighbors.data[i]->nodeData.cumulativeValue ) SetFunctionCumulative( pNeighbors.neighbors.data[i] );
						cumV += pNeighbors.neighbors.data[i]->nodeData.v * weight;
					}
				}
				node->nodeData.v += cumV;
			}
			node->nodeData.cumulativeValue = true;
		};

	typename ElementFunctionNeighborKey::NeighborType &neighbors = _efKey.neighbors[ node->depth() ];
	for( unsigned int i=0 ; i<Window::IsotropicSize< Dim , 2 >() ; i++ )
	{
#ifdef DEBUG_ISO_TREE
		if( !neighbors.neighbors.data[i] ) MK_ERROR_OUT( "Expected valid function node" );
#endif // DEBUG_ISO_TREE
		if( !neighbors.neighbors.data[i]->nodeData.cumulativeValue ) SetFunctionCumulative( neighbors.neighbors.data[i] );
	}
	node->nodeData.validElement = true;
}

template< unsigned int Dim , unsigned int CoDim >
const typename IsoTree< Dim , CoDim >::Node * IsoTree< Dim , CoDim >::_getNode( Hat::Index< Dim > I , unsigned int depth ) const
{
	unsigned int res = 1<<(depth+1);
	Point< double , Dim > p;
	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( I[d] + 0.5 ) / res;
	return _root.getNode( p , depth+1 );
}

template< unsigned int Dim , unsigned int CoDim >
typename IsoTree< Dim , CoDim >::Node * IsoTree< Dim , CoDim >::_getNode( Hat::Index< Dim > I , unsigned int depth )
{
	unsigned int res = 1<<(depth+1);
	Point< double , Dim > p;
	for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = ( I[d] + 0.5 ) / res;
	return _root.getNode( p , depth+1 );
}

