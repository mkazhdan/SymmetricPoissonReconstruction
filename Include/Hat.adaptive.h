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

#ifndef HAT_ADAPTIVE_INCLUDED
#define HAT_ADAPTIVE_INCLUDED

// [NOTE]
// To support prolongation, we add a one-ring-neighborhood of padding arund the "active" functions so
// we can represent the prolongation of coarser functions that can be "seen" by the active functions.

// An abstract class representing:
// 1. A set of active function indices
// 2. The set of elements supported on the active functions
// 3. The set of functions whose support overlaps the active functions
// [INVARIANTS]
// 1. All elements in the support of active functions exist
// 2. All functions supported on elements exist
template< unsigned int Dim >
struct BaseIndexer
{
	/////////////////////
	// Virtual Methods //
	/////////////////////

	// The resolution of the underlying grid
	virtual size_t resolution( void ) const = 0;

	// The number of indexed functions
	virtual size_t numFunctions( void ) const = 0;

	// The number of active indexed functions
	virtual size_t numActiveFunctions( void ) const { return numFunctions(); }

	// The number of elements supported by active functions
	virtual size_t numElements( void ) const = 0;

	// The Dim-dimensional coordinates of the f-th function
	virtual Hat::Index< Dim > functionIndex( size_t f ) const = 0;

	// The Dim-dimensional coordinates of the e-th function
	virtual Hat::Index< Dim > elementIndex( size_t e ) const = 0;

	// The indices of the functions supported on the element
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors( size_t e , unsigned int thread ) const = 0;

	// The indices of the elements supported on the function
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors( size_t f , unsigned int thread ) const = 0;

	// The indices of the functions in the one-ring-neighborhood
	virtual Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors( size_t f , unsigned int thread ) const;

	// The indices of the functions supported on the element containing the point
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors( Point< double , Dim > p , unsigned int thread ) const = 0;

	// The indices of the functions in the one-ring-neighborhood and the elements supported on the function
	virtual std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > fNeighbors( size_t f , unsigned int thread ) const;
};

// [INVARIANTS]
// 1. Parent element exists
template< unsigned int Dim >
struct BaseProlongationIndexer : public BaseIndexer< Dim >
{
	/////////////////////
	// Virtual Methods //
	/////////////////////

	// The indices of the functions in the one-ring-neighborhood of the child
	virtual Window::IsotropicStaticWindow< size_t , Dim , 3 > ffChildNeighbors( size_t f , unsigned int thread ) const = 0;

	// Returns the parent of an element
	virtual size_t parentElement( size_t e , unsigned int thread ) const = 0;

	// Returns the children of an element
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > childElements( size_t e , unsigned int thread ) const = 0;

};

// An abstract class representing a hierarchy:
// 1. A set of active function indices
// 2. The set of elements supported on the active functions
// 3. The set of functions whose support overlaps the active functions
template< unsigned int Dim >
struct BaseHierarchicalIndexer
{
	/////////////////////
	// Virtual Methods //
	/////////////////////

	// The maximum depth
	virtual unsigned int maxDepth( void ) const = 0;

	// The resolution of the underlying grid
	virtual size_t resolution( unsigned int depth ) const = 0;

	// The number of indexed functions
	virtual size_t numFunctions( unsigned int depth ) const = 0;

	// The number of active indexed functions
	virtual size_t numActiveFunctions( unsigned int depth ) const { return numFunctions( depth ); }

	// The number of elements supported by active functions
	virtual size_t numElements( unsigned int depth ) const = 0;

	// The Dim-dimensional coordinates of the f-th function
	virtual Hat::Index< Dim > functionIndex( unsigned int depth , size_t f ) const = 0;

	// The Dim-dimensional coordinates of the e-th function
	virtual Hat::Index< Dim > elementIndex( unsigned int depth , size_t e ) const = 0;

	// The indices of the functions supported on the element
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors( unsigned int depth , size_t e , unsigned int thread ) const = 0;

	// The indices of the elements supported on the function
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors( unsigned int depth , size_t f , unsigned int thread ) const = 0;

	// The indices of the functions in the one-ring-neighborhood
	virtual Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors( unsigned int depth , size_t f , unsigned int thread ) const;

	// The indices of the functions in the one-ring-neighborhood of the child
	virtual Window::IsotropicStaticWindow< size_t , Dim , 3 > ffChildNeighbors( unsigned int depth , size_t f , unsigned int thread ) const = 0;

	// The indices of the functions supported on the element containing the point
	virtual Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors( unsigned int depth , Point< double , Dim > p , unsigned int thread ) const = 0;

	// The indices of the functions supported on the element containing the point
	virtual std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > fNeighbors( unsigned int depth , size_t f , unsigned int thread ) const;

	// Returns the parent of an element
	size_t parentElement( unsigned int depth , size_t e , unsigned int thread ) const;

	// Returns the children of an element
	Window::IsotropicStaticWindow< size_t , Dim , 2 > childElements( unsigned int depth , size_t e , unsigned int thread ) const;

	// Outputs the hierarchical indexer to a file
	virtual void write( BinaryStream &stream , bool serialize ) const = 0;

	/////////////////////////
	// Implemented Methods //
	/////////////////////////
	Eigen::VectorXd regularCoefficients( const std::vector< Eigen::VectorXd > &hierarchicalCoefficients ) const;
};

// A class for extracting individual, per-depth, indexers from a hierarchy
template< unsigned int Dim , typename HierarchyType >
struct Indexer : public BaseProlongationIndexer< Dim >
{
	Indexer( const HierarchyType &hierarchy , unsigned int depth ) : _hierarchy( &hierarchy ) , _depth( depth ) {}
	size_t resolution( void ) const { return _hierarchy->resolution( _depth ); }
	size_t numFunctions( void ) const { return _hierarchy->numFunctions( _depth ); }
	size_t numActiveFunctions( void ) const { return _hierarchy->numActiveFunctions( _depth ); }
	size_t numElements( void ) const { return _hierarchy->numElements( _depth ); }
	Hat::Index< Dim > functionIndex( size_t f ) const { return _hierarchy->functionIndex( _depth , f ); }
	Hat::Index< Dim > elementIndex( size_t e ) const { return _hierarchy->elementIndex( _depth , e ); }
	Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors( size_t e , unsigned int thread ) const { return _hierarchy->efNeighbors( _depth , e , thread ); }
	Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors( size_t f , unsigned int thread ) const { return _hierarchy->feNeighbors( _depth , f , thread ); }
	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors( size_t f , unsigned int thread ) const { return _hierarchy->ffNeighbors( _depth , f , thread ); }
	Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors( Point< double , Dim > p , unsigned int thread ) const { return _hierarchy->fNeighbors( _depth , p , thread ); }
	std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > fNeighbors( size_t f , unsigned int thread ) const { return _hierarchy->fNeighbors( _depth , f , thread ); }
	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffChildNeighbors( size_t f , unsigned int thread ) const { return _hierarchy->ffChildNeighbors( _depth , f , thread ); }
	size_t parentElement( size_t e , unsigned int thread ) const { return _hierarchy->parentElement( _depth , e , thread ); }
	Window::IsotropicStaticWindow< size_t , Dim , 2 > childElements( size_t e , unsigned int thread ) const { return _hierarchy->childElements( _depth , e , thread ); }

protected:
	const HierarchyType * _hierarchy;
	unsigned int _depth;
};

template< unsigned int Dim >
struct HierarchicalRegularIndexer : public BaseHierarchicalIndexer< Dim >
{
	using BaseHierarchicalIndexer< Dim >::fNeighbors;
	using Indexer = Hat::Indexer< Dim , HierarchicalRegularIndexer< Dim > >;
	Indexer operator[]( unsigned int depth ) const { return Indexer( *this , depth ); }

	HierarchicalRegularIndexer( unsigned int maxDepth );
	HierarchicalRegularIndexer( BinaryStream &stream );

	unsigned int maxDepth( void ) const;

	size_t resolution( unsigned int depth ) const;

	size_t numFunctions( unsigned int depth ) const;

	size_t numActiveFunctions( unsigned int depth ) const;

	size_t numElements( unsigned int depth ) const;

	Hat::Index< Dim > functionIndex( unsigned int depth , size_t f ) const;

	Hat::Index< Dim > elementIndex( unsigned int depth , size_t e ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors( unsigned int depth , size_t f , unsigned int thread=0 ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffChildNeighbors( unsigned int depth , size_t f , unsigned int thread=0 ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors( unsigned int depth , size_t e , unsigned int thread ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors( unsigned int depth , size_t e , unsigned int thread ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors( unsigned int depth , Point< double , Dim > p , unsigned int thread ) const;

	size_t parentElement( unsigned int depth , size_t e , unsigned int thread=0 ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > childElements( unsigned int depth , size_t e , unsigned int thread=0 ) const;

	void write( BinaryStream &stream , bool serialize=false ) const;

protected:
	std::vector< Hat::ScalarFunctions< Dim > > _scalarFunctions;
};

template< unsigned int Dim >
struct HierarchicalAdaptedIndexer : public BaseHierarchicalIndexer< Dim >
{
	using Indexer = Hat::Indexer< Dim , HierarchicalAdaptedIndexer< Dim > >;

	// The information stored within a node of the tree
	struct NodeData
	{
		size_t functionIndex , elementIndex;
		NodeData( void ) : functionIndex(-1) , elementIndex(-1){}
	};

	// The tree node itself
	using Node = RegularTreeNode< Dim , NodeData , unsigned short >;

	Indexer operator[]( unsigned int depth ) const { return Indexer( *this , depth ); }

	template< typename SampleFunctor /* = std::function< Point< double , Dim > ( unsigned int idx ) > */ >
	HierarchicalAdaptedIndexer( size_t sampleNum , SampleFunctor && Samples , unsigned int kernelRadius , unsigned int maxDepth );
	
	HierarchicalAdaptedIndexer( BinaryStream &stream );

	unsigned int maxDepth( void ) const;

	size_t resolution( unsigned int depth ) const;
	
	size_t numFunctions( unsigned int depth ) const;

	size_t numActiveFunctions( unsigned int depth ) const;

	size_t numElements( unsigned int depth ) const;

	Hat::Index< Dim > functionIndex( unsigned int depth , size_t f ) const;

	Hat::Index< Dim > elementIndex( unsigned int depth , size_t e ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffNeighbors( unsigned int depth , size_t f , unsigned int thread ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 3 > ffChildNeighbors( unsigned int depth , size_t f , unsigned int thread ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > efNeighbors( unsigned int depth , size_t e , unsigned int thread ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > feNeighbors( unsigned int depth , size_t f , unsigned int thread ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > fNeighbors( unsigned int depth , Point< double , Dim > p , unsigned int thread ) const;

	std::pair< Window::IsotropicStaticWindow< size_t , Dim , 3 > , Window::IsotropicStaticWindow< size_t , Dim , 2 > > fNeighbors( unsigned int depth , size_t f , unsigned int thread ) const;

	size_t parentElement( unsigned int depth , size_t e , unsigned int thread=0 ) const;

	Window::IsotropicStaticWindow< size_t , Dim , 2 > childElements( unsigned int depth , size_t e , unsigned int thread=0 ) const;

	void write( BinaryStream &stream , bool serialize=false ) const;

	const Node & root( void ) const;
	
protected:

	// A key for finding the the functions incident on a cell
	template< unsigned int KernelRadius=0 >
	using ElementFunctionNeighborKey = typename Node::template NeighborKey< ParameterPack::IsotropicUIntPack< Dim , KernelRadius > , ParameterPack::IsotropicUIntPack< Dim , KernelRadius+1 > >;

	// A key for funding the functions incident on an element
	using ConstElementFunctionNeighborKey = typename Node::template ConstNeighborKey< ParameterPack::IsotropicUIntPack< Dim , 0 > , ParameterPack::IsotropicUIntPack< Dim , 1 > >;

	// A key for funding the functions incident on an element
	using ConstFunctionElementNeighborKey = typename Node::template ConstNeighborKey< ParameterPack::IsotropicUIntPack< Dim , 1 > , ParameterPack::IsotropicUIntPack< Dim , 0 > >;

	// A key for funding the functions incident on a function
	using ConstFunctionFunctionNeighborKey = typename Node::template ConstNeighborKey< ParameterPack::IsotropicUIntPack< Dim , 1 > , ParameterPack::IsotropicUIntPack< Dim , 1 > >;

	/////////////////
	// Member data //
	Node _root , *_spaceRoot;
	std::vector< std::vector< Node * > > _functionNodes;
	std::vector< std::vector< Node * > > _elementNodes;
	std::vector< size_t > _activeSizes;
	std::vector< ConstFunctionFunctionNeighborKey > _ffKeys;
	std::vector< ConstElementFunctionNeighborKey > _efKeys;
	std::vector< ConstFunctionElementNeighborKey > _feKeys;
	/////////////////

	NodeData & _functionNodeData( unsigned int depth , size_t i );
	const NodeData & _functionNodeData( unsigned int depth , size_t i ) const;

	NodeData & _elementNodeData( unsigned int depth , size_t i );
	const NodeData & _elementNodeData( unsigned int depth , size_t i ) const;

	template< unsigned int KernelRadius , typename SampleFunctor /* = std::function< Point< double , Dim > ( size_t idx ) > */ >
	void _addSamples( size_t sampleNum , SampleFunctor && Samples , unsigned int maxDepth );

	void _finalize( unsigned int maxDepth );
};

#include "Hat.adaptive.inl"

#endif // HAT_ADAPTIVE_INCLUDED