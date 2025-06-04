#ifndef KDTREE_H
#define KDTREE_H
/*
Szymon Rusinkiewicz
Princeton University

KDtree.h
A K-D tree for points, with limited capabilities (find points within a radius of a given point). 
*/

#include <vector>
#include <functional>
#include "Misha/Geometry.h"
#include "Misha/Array.h"
#include "Mempool.h"

namespace MishaK
{
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE=7 >
	class KDTree
	{
		class Node;
		Node *_root;
		double _radius;
	public:
		// Constructor from an array of points
		KDTree( const std::function< Point< double , D > (unsigned int) > &pFunction , unsigned int n );
		~KDTree( void );

		// The queries: returns the set of points that are within sqrt(dist2)
		void add_nearest_neighbors( Point< double , D > p , double dist2 , std::vector< std::pair< unsigned int , Point< double , D > > > &neighbors ) const;

		std::vector< std::pair< unsigned int , Point< double , D > > > get_k_nearest_neighbors( Point< double , D > p , unsigned int k ) const { return get_k_nearest_neighbors( p , k , radius()/1000. ); }
		std::vector< std::pair< unsigned int , Point< double , D > > > get_k_nearest_neighbors( Point< double , D > p , unsigned int k , double r ) const;

		double radius( void ) const { return _radius; }
	};


#include <cmath>
#include <string.h>
#include "Mempool.h"
#include <vector>
#include <algorithm>

	// Class for nodes in the K-D tree
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	class KDTree< D , MAX_PTS_PER_NODE >::Node
	{
	private:
		static PoolAlloc memPool;

	public:
		// The node itself

		unsigned int npts; // If this is 0, intermediate node.  If nonzero, leaf.

		union
		{
			struct
			{
				Point< double , D > center;
				double r;
				unsigned int splitaxis;
				Node *child1 , *child2;
			} node;
			struct
			{
				std::pair< unsigned int , Point< double , D > > p[MAX_PTS_PER_NODE];
			} leaf;
		};

		Node( Pointer( std::pair< unsigned int , Point< double , D > > ) pts , unsigned int n );
		~Node();

		void add_nearest_neighbors( Point< double , D > p , double dist , std::vector< std::pair< unsigned int , Point< double , D > > > &neighbors ) const;

		void *operator new( size_t n ) { return memPool.alloc(n); }
		void operator delete( void *p , size_t n ) { memPool.free(p,n); }
	};


	// Class static variable
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE > PoolAlloc KDTree< D , MAX_PTS_PER_NODE >::Node::memPool( sizeof(KDTree::Node) );


	// Create a KD tree from the points pointed to by the array pts
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	KDTree< D , MAX_PTS_PER_NODE >::Node::Node( Pointer( std::pair< unsigned int , Point< double , D > > ) p , unsigned int n )
	{
		// Leaf nodes
		if( n<=MAX_PTS_PER_NODE )
		{
			npts = n;
			memcpy( leaf.p , p , sizeof( std::pair< unsigned int , Point< double , D > > )*n );
			return;
		}

		// Else, interior nodes
		npts = 0;
		// Find bbox
		Point< double , D > min , max;
		min = max = p[0].second;
		for( unsigned int i=1 ; i<n ; i++ ) for( unsigned int d=0 ; d<D ; d++ )
		{
			min[d] = std::min< double >( min[d] , p[i].second[d] );
			max[d] = std::max< double >( max[d] , p[i].second[d] );
		}

		// Find node center and size
		node.center = ( min + max ) * 0.5;
		Point< double , D > d = max-min;
		node.r = Point< double , D >::Length( d/2 );

		// Find longest axis
		node.splitaxis = D-1;
		for( int i=0 ; i<D-1 ; i++ ) if( d[i]>d[node.splitaxis] ) node.splitaxis = i;

		// Partition
		const double splitval = node.center[ node.splitaxis ];
		int l=0 , r=n-1;
		while( l<=r )
		{
			while( p[l].second[node.splitaxis]< splitval && l<(int)n ) l++;
			while( p[r].second[node.splitaxis]>=splitval && r>=0 ) r--;
			if( l>r ) break;
			std::swap( p[l] , p[r] );
			l++ , r--;
		}

		// Check for bad cases of clustered points
		if ( l==0 || l==n ) l = n/2;

		// Build subtrees
		node.child1 = new Node( p , l );
		node.child2 = new Node( p+l , n-l );
	}


	// Destroy a KD tree node
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	KDTree< D , MAX_PTS_PER_NODE >::Node::~Node()
	{
		if( !npts )
		{
			delete node.child1;
			delete node.child2;
		}
	}


	// Crawl the KD tree
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	void KDTree< D , MAX_PTS_PER_NODE >::Node::add_nearest_neighbors( Point< double , D > p , double dist , std::vector< std::pair< unsigned int , Point< double , D > > > &neighbors ) const
	{
		// Leaf nodes
		if( npts )
		{
			for( unsigned int i=0 ; i<npts ; i++ ) if( Point< double , D >::SquareNorm( leaf.p[i].second - p )<=dist*dist ) neighbors.push_back( leaf.p[i] );
			return;
		}

		// Check whether to abort
		if( Point< double , D >::SquareNorm( node.center - p )>=(node.r+dist)*(node.r+dist) ) return;
		else
		{
			node.child1->add_nearest_neighbors( p , dist , neighbors );
			node.child2->add_nearest_neighbors( p , dist , neighbors );
		}
	}


	// Create a KDTree from a list of points (i.e., ptlist is a list of D*n floats)
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	KDTree< D , MAX_PTS_PER_NODE >::KDTree( const std::function< Point< double , D > (unsigned int) > &pFunction , unsigned int n )
	{
		std::vector< std::pair< unsigned int , Point< double , D > > > pts(n);
		for( unsigned int i=0 ; i<n ; i++ ) pts[i] = std::make_pair( i , pFunction(i) );
		_root = new Node( GetPointer( pts ) , n );
		_radius = 0;
		if( _root->npts )
		{
			Point< double , D >  min , max;
			min = max = _root->leaf.p[0].second;
			for( unsigned int i=1 ; i<_root->npts ; i++ ) for( unsigned int d=0 ; d<D ; d++ )
			{
				min[d] = std::min< double >( min[d] , _root->leaf.p[i].second[d] );
				max[d] = std::max< double >( max[d] , _root->leaf.p[i].second[d] );
			}
			_radius = Point< double , D >::Length( max-min ) / 2;
		}
		else _radius = _root->node.r;
	}


	// Delete a KDTree
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE > KDTree< D , MAX_PTS_PER_NODE >::~KDTree( void ) { delete _root; }


	// Return the closest point in the KD tree to p
	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	void KDTree< D , MAX_PTS_PER_NODE >::add_nearest_neighbors( Point< double , D > p , double dist , std::vector< std::pair< unsigned int , Point< double , D > > > &neighbors ) const
	{
		if( dist <= 0.0f ) dist = _radius;
		_root->add_nearest_neighbors( p , dist , neighbors );
	}

	template< unsigned int D , unsigned int MAX_PTS_PER_NODE >
	std::vector< std::pair< unsigned int , Point< double , D > > > KDTree< D , MAX_PTS_PER_NODE >::get_k_nearest_neighbors( Point< double , D > p , unsigned int k , double r ) const
	{
		std::vector< std::pair< unsigned int , Point< double , D > > > neighbors;
		if( r<=0 )
		{
			fprintf( stderr , "[ERROR] Expected positive radius: %g\n" , r );
			exit( 0 );
		}
		neighbors.reserve( k );

		auto Compare = [&]( const std::pair< unsigned int , Point< double , D > > &p1 , const std::pair< unsigned int , Point< double , D > > &p2 )
			{
				return Point< double , D >::SquareNorm( p-p1.second ) < Point< double , D >::SquareNorm( p-p2.second );
			};
		while( true )
		{
			add_nearest_neighbors( p , r , neighbors );
			if( neighbors.size()>=k )
			{
				std::sort( neighbors.begin() , neighbors.end() , Compare );
				neighbors.resize( k );
				return neighbors;
			}
			else if( r>=radius() )
			{
				std::sort( neighbors.begin() , neighbors.end() , Compare );
				return neighbors;
			}
			else
			{
				r *= 2;
				neighbors.resize( 0 );
			}
		}
		std::cerr << "[ERROR] Should not be here" << std::endl;
		exit( 0 );
		return neighbors;
	}
}
#endif
