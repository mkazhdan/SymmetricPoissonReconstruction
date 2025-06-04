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

#ifndef ISO_TREE_INCLUDED
#define ISO_TREE_INCLUDED

#include "Eigen/Dense"
#include "Misha/RegularTree.h"


namespace MishaK
{
	template< unsigned int Dim , unsigned int CoDim >
	struct IsoTree
	{
		// The data stored with the tree node
		struct NodeData
		{
			NodeData( void ) : validElement(false) , cumulativeValue(false) , state(0){}
			Point< double , CoDim > v;
			bool validElement , cumulativeValue;
			unsigned char state;
		};

		// The tree node itself
		using Node = RegularTreeNode< Dim , NodeData , unsigned short >;

		IsoTree( void );

		template< typename HierarchicalIndexer /* = Hat::BaseHierarchicalIndexer< Dim > */ , typename CoefficientFunctor /* = std::function< Point< double , CoDim > (unsigned int , size_t ) > */ >
		IsoTree( const HierarchicalIndexer & indexer , CoefficientFunctor && F , unsigned int maxDepth  );

		template< typename HierarchicalIndexer /* = Hat::BaseHierarchicalIndexer< Dim > */ , typename CoefficientFunctor /* = std::function< Point< double , CoDim > (unsigned int , size_t ) > */ >
		IsoTree( const HierarchicalIndexer & indexer , CoefficientFunctor && F );

		// Adds an element into the tree and ensures that associated cumulative values are set
		Node * addElement( Hat::Index< Dim > E , unsigned int depth );

		// Adds the neighbors of an element node into the tree
		template< typename AddFunctor = std::function< bool ( Node * ) > >
		Window::IsotropicStaticWindow< typename IsoTree< Dim , CoDim >::Node * , Dim , 3 > addElementNeighbors( Node * node , AddFunctor && F=[]( Node * ){ return true; } );

		// Returns the values at the corners of the element
		Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > elementValues( const Node * node , unsigned int thread ) const;
		Window::IsotropicStaticWindow< Point< double , CoDim > , Dim , 2 > elementValues( Hat::Index< Dim > E , unsigned int thread ) const;

		// Returns the value at the function
		Point< double , CoDim > functionValue( Hat::Index< Dim > F ) const;

		Node &spaceRoot( void );
		const Node &spaceRoot( void ) const;

		Point< double , CoDim > value( Point< double , Dim > p , unsigned int thread , bool forceMaxDepth=false ) const;
		Point< Point< double , Dim > , CoDim , double > gradient( Point< double , Dim > p , unsigned int thread , bool forceMaxDepth=false ) const;

		bool isZeroCrossingNode( const Node * node , bool all , unsigned int thread ) const;

		void refineZeroCrossing( bool all );
		std::vector< const Node * > levelSetNodes( bool all ) const;
	protected:
		// Keys for funding the functions incident on an element
		using       ElementElementNeighborKey = typename Node::template      NeighborKey< ParameterPack::IsotropicUIntPack< Dim , 1 > , ParameterPack::IsotropicUIntPack< Dim , 1 > >;
		using      ElementFunctionNeighborKey = typename Node::template      NeighborKey< ParameterPack::IsotropicUIntPack< Dim , 0 > , ParameterPack::IsotropicUIntPack< Dim , 1 > >;
		using ConstElementFunctionNeighborKey = typename Node::template ConstNeighborKey< ParameterPack::IsotropicUIntPack< Dim , 0 > , ParameterPack::IsotropicUIntPack< Dim , 1 > >;


		/////////////////
		// Member data //
		unsigned int _maxDepth;
		Node _root , *_spaceRoot;
		ElementElementNeighborKey _eeKey;
		ElementFunctionNeighborKey _efKey;
		std::vector< ConstElementFunctionNeighborKey > _efKeys;
		/////////////////

		IsoTree( unsigned int maxDepth );
		Node * _addFunction( Hat::Index< Dim > F , unsigned int depth );
		Node * _addElement( Hat::Index< Dim > E , unsigned int depth );
		void _setElementCumulative( Node * node );
		const Node * _getNode( Hat::Index< Dim > I , unsigned int depth ) const;
		Node * _getNode( Hat::Index< Dim > I , unsigned int depth );

	};

#include "IsoTree.inl"
}

#endif // ISO_TREE_INCLUDED