#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include "Misha/Miscellany.h"
#include "Misha/CmdLineParser.h"
#include "Misha/Geometry.h"
#include "Misha/MarchingSimplices.h"
#include "Misha/Ply.h"
#include "Misha/PlyVertexData.h"
#include "Include/PreProcessor.h"
#include "Include/UnsignedNormals.h"
#include "Include/Samples.h"
#include "SymmetricPoissonRecon.h"
#include "Include/IsoTree.h"
#include "Misha/IsoSurface3D.h"


using namespace MishaK;

static const unsigned int CoDim = 1;
static const bool Sym = true;

CmdLineParameter< std::string >
	In( "in" ) ,
	Out( "out" );

CmdLineParameter< double >
	Scale( "scale" , 1.25 ) ,
	ScreeningWeight( "sWeight" , 5e4 ) ,
	BoundaryStiffnessWeight( "dWeight" , 1e2 ) ,
	IterationMultiplier( "iMult" , 2. ) ,
	SamplesPerNode( "samplesPerNode" , 2. );

CmdLineParameter< unsigned int >
	Depth( "depth" , 7 ) ,
	SolveDepth( "solveDepth" ) ,
	ExtractDepth( "extractDepth" ) ,
	KernelDepth( "kDepth" ) ,
	TensorKernelRadius( "tRadius" , 1 ) ,
	DensityKernelRadius( "dRadius" , 1 ) ,
	Verbosity( "verbose" , 0 ) ,
	MinDepth( "minDepth" , 3 ) ,
	CoarseIterations( "coarseIters" , 512 ) ,
	Iterations( "iters" , 10 ) ,
	AdaptiveDilationRadius( "dilationRadius" , 3 ) ,
#ifdef USE_NL_OPT
	GradientMemory( "memory" , -1 ) ,
#endif // USE_NL_OPT
	NearestNeighbors( "nn" , 20 );

CmdLineReadable
	SingleLevel( "singleLevel" ) ,
	NoSign( "noSign" ) ,
	Triangulate( "triangulate" ) ,
	FirstOrder( "firstOrder" );

CmdLineReadable* params[] =
{
	&In ,
	&Out ,
	&Depth ,
	&SolveDepth ,
	&ExtractDepth ,
	&KernelDepth ,
	&DensityKernelRadius ,
	&TensorKernelRadius ,
	&CoarseIterations ,
	&Iterations ,
	&Verbosity ,
	&ScreeningWeight ,
	&BoundaryStiffnessWeight ,
	&SingleLevel ,
	&Scale ,
	&IterationMultiplier ,
	&MinDepth ,
	&Triangulate ,
	&NearestNeighbors ,
	&AdaptiveDilationRadius ,
	&NoSign ,
#ifdef USE_NL_OPT
	&GradientMemory ,
#endif // USE_NL_OPT
	&FirstOrder ,
	&SamplesPerNode ,
	NULL
};

void ShowUsage( const char* ex )
{
	printf( "Usage %s:\n" , ex );
	printf( "\t --%s <oriented input points>\n" , In.name.c_str() );
	printf( "\t[--%s <output grid/mesh>]\n" , Out.name.c_str() );
	printf( "\t[--%s <grid depth>=%d]\n" , Depth.name.c_str() , Depth.value );
	printf( "\t[--%s <solver depth>]\n" , SolveDepth.name.c_str() );
	printf( "\t[--%s <extraction depth>]\n" , ExtractDepth.name.c_str() );
	printf( "\t[--%s <kernel depth>=depth]\n" , KernelDepth.name.c_str() );
	printf( "\t[--%s <minimum hierarchy solver depth>=%d]\n" , MinDepth.name.c_str() , MinDepth.value );
	printf( "\t[--%s <density kernel radius>=%d]\n" , DensityKernelRadius.name.c_str() , DensityKernelRadius.value );
	printf( "\t[--%s <tensor kernel radius>=%d]\n" , TensorKernelRadius.name.c_str() , TensorKernelRadius.value );
	printf( "\t[--%s <coarse solver iterations>=%d]\n" , CoarseIterations.name.c_str() , CoarseIterations.value );
	printf( "\t[--%s <solver iterations>=%d]\n" , Iterations.name.c_str() , Iterations.value );
	printf( "\t[--%s <iteration multiplier>=%f]\n" , IterationMultiplier.name.c_str() , IterationMultiplier.value );
	printf( "\t[--%s <screening weight>=%g]\n" , ScreeningWeight.name.c_str() , ScreeningWeight.value );
	printf( "\t[--%s <boundary stiffness weight>=%g]\n" , BoundaryStiffnessWeight.name.c_str() , BoundaryStiffnessWeight.value );
	printf( "\t[--%s <bounding box scale>=%f]\n" , Scale.name.c_str() , Scale.value );
	printf( "\t[--%s <nearest neighbors for normal fitting>=%d]\n" , NearestNeighbors.name.c_str() , NearestNeighbors.value );
	printf( "\t[--%s <adaptive dilation radius>=%d]\n" , AdaptiveDilationRadius.name.c_str() , AdaptiveDilationRadius.value );
#ifdef USE_NL_OPT
	printf( "\t[--%s <gradient memory>=%d]\n" , GradientMemory.name.c_str() , GradientMemory.value );
#endif // USE_NL_OPT
	printf( "\t[--%s <number of samples per node>=%f]\n" , SamplesPerNode.name.c_str() , SamplesPerNode.value );
	printf( "\t[--%s]\n" , SingleLevel.name.c_str() );
	printf( "\t[--%s]\n" , NoSign.name.c_str() );
	printf( "\t[--%s]\n" , Triangulate.name.c_str() );
	printf( "\t[--%s]\n" , FirstOrder.name.c_str() );
	printf( "\t[--%s <verbosity>=%d]\n" , Verbosity.name.c_str() , Verbosity.value );
}

template< unsigned int Dim , typename Energy , typename EstimatorReal , typename SampleData , typename HierarchicalIndexer >
void Execute( const std::vector< std::pair< Point< double , Dim > , SampleData > > & sampleData , const HierarchicalIndexer & hierarchicalIndexer , SquareMatrix< double , Dim+1 > unitCubeToWorld );

template< unsigned int Dim , typename EstimatorReal , typename SampleData >
void Execute( std::vector< std::pair< Point< double , Dim > , SampleData > > sampleData );

template< unsigned int Dim >
void Execute( std::string fileName );

template< unsigned int Dim >
std::vector< std::pair< Point< double , Dim > , Hat::SquareMatrix< double , Dim , Sym > > > ReadSamples( std::string fileName , bool verbose );

int main( int argc , char* argv[] )
{
	CmdLineParse( argc-1 , argv+1 , params );
	if( !In.set )
	{
		ShowUsage( argv[0] );
		return EXIT_SUCCESS;
	}

	if( !KernelDepth.set ) KernelDepth.value = Depth.value;
	if( KernelDepth.value>Depth.value )
	{
		MK_WARN( "Kernel depth exceeds depth: " , KernelDepth.value , " <- " , Depth.value );
		KernelDepth.value = Depth.value;
	}

	Miscellany::PerformanceMeter::Width = 18;
	Miscellany::PerformanceMeter pMeter( '.' );

	Execute< 3 >( In.value );

	std::cout << pMeter( "Performance" ) << std::endl;

	return EXIT_SUCCESS;
}

template< unsigned int Dim >
void Execute( std::string fileName )
{
	using EstimatorReal = float;

	using SampleData = Hat::SquareMatrix< double , Dim , Sym >;
	std::vector< std::pair< Point< double , Dim > , SampleData > > sampleData;
	{
		Miscellany::PerformanceMeter pMeter( '.' );
		sampleData = ReadSamples< Dim >( fileName , Verbosity.value );
		if( Verbosity.value ) std::cout << pMeter( "Read samples" ) << std::endl;
	}
	Execute< Dim , EstimatorReal >( sampleData );
}

template< unsigned int Dim , typename EstimatorReal , typename SampleData >
void Execute( std::vector< std::pair< Point< double , Dim > , SampleData > > sampleData )
{
	Miscellany::PerformanceMeter *pMeter = new Miscellany::PerformanceMeter();
	SquareMatrix< double , Dim+1 > unitCubeToWorld;
	{
		SquareMatrix< double , Dim+1 > worldToUnitCube = SymPR::ToUnitCube< Dim >( [&]( size_t idx ){ return sampleData[idx].first; } , sampleData.size() , Scale.value );
		double scale = pow( pow( worldToUnitCube.determinant() , 1./Dim ) , Dim-CoDim );
		for( unsigned int i=0 ; i<sampleData.size() ; i++ ) sampleData[i].first = worldToUnitCube( sampleData[i].first ) , sampleData[i].second *= scale;
		unitCubeToWorld = worldToUnitCube.inverse();
	}

	if( AdaptiveDilationRadius.value==-1 )
	{
		using HierarchicalIndexer = Hat::HierarchicalRegularIndexer< Dim >;
		HierarchicalIndexer hierarchicalIndexer( Depth.value );
		if( Verbosity.value && Verbosity.value!=-1 ) std::cout << (*pMeter)( "Indexer" ) << std::endl;
		delete pMeter;
		return Execute< Dim , ProductSystem::CascadicSystemEnergy< Dim , Sym , typename HierarchicalIndexer::Indexer > , EstimatorReal >( sampleData , hierarchicalIndexer , unitCubeToWorld );
	}
	else
	{
		using HierarchicalIndexer = Hat::HierarchicalAdaptedIndexer< Dim >;
		HierarchicalIndexer hierarchicalIndexer( sampleData.size() , [&]( size_t i ){ return sampleData[i].first; } , AdaptiveDilationRadius.value , Depth.value );
		if( Verbosity.value && Verbosity.value!=-1 ) std::cout << (*pMeter)( "Indexer" ) << std::endl;
		delete pMeter;
		return Execute< Dim , ProductSystem::CascadicSystemEnergy< Dim , Sym , typename HierarchicalIndexer::Indexer > , EstimatorReal >( sampleData , hierarchicalIndexer , unitCubeToWorld );
	}
}

template< unsigned int Dim , typename Energy , typename EstimatorReal , typename SampleData , typename HierarchicalIndexer >
void Execute( const std::vector< std::pair< Point< double , Dim > , SampleData > > & sampleData , const HierarchicalIndexer & hierarchicalIndexer , SquareMatrix< double , Dim+1 > unitCubeToWorld )
{
	Miscellany::PerformanceMeter pMeter( '.' );

	MinDepth.value = std::min< unsigned int >( MinDepth.value , Depth.value );

	if( !SolveDepth.set ) SolveDepth.value = Depth.value;
	else
	{
		if( SolveDepth.value<MinDepth.value )
		{
			MK_WARN( "Solve depth less than min depth: " , SolveDepth.value , " -> " , MinDepth.value );
			SolveDepth.value = MinDepth.value;
		}
		if( SolveDepth.value>Depth.value )
		{
			MK_WARN( "Solve depth greater than depth: " , SolveDepth.value , " -> " , Depth.value );
			SolveDepth.value = Depth.value;
		}
	}

	if( !ExtractDepth.set ) ExtractDepth.value = SolveDepth.value;
	else
	{
		if( ExtractDepth.value<MinDepth.value )
		{
			MK_WARN( "Extraction depth less than min depth: " , ExtractDepth.value , " -> " , MinDepth.value );
			ExtractDepth.value = MinDepth.value;
		}
		if( ExtractDepth.value>Depth.value )
		{
			MK_WARN( "Extraction depth greater than depth: " , ExtractDepth.value , " -> " , Depth.value );
			ExtractDepth.value = Depth.value;
		}
	}

	// Splat the samples in
	srand( 0 );

	Eigen::VectorXd solution;
	{
		Hat::ScalarFunctions< Dim > scalars( 1<<MinDepth.value );
		solution.resize( hierarchicalIndexer[MinDepth.value].numFunctions() );
		{
			Hat::Range< Dim > range;
			Point< double , Dim > c;
			for( unsigned int d=0 ; d<Dim ; d++ ) range.second[d] = (1<<MinDepth.value)+1 , c[d] = ( 1<<MinDepth.value )/2.;
			for( unsigned int i=0 ; i<solution.size() ; i++ )
			{
				Hat::Index< Dim > F = hierarchicalIndexer[MinDepth.value].functionIndex(i);
				Point< double , Dim > p;
				for( unsigned int d=0 ; d<Dim ; d++ ) p[d] = F[d];
				solution[i] = Point< double , Dim >::SquareNorm( p - c );
			}
		}
	}

	OrderedSampler< Dim > orderedSampler( [&]( size_t i ){ return sampleData[i].first; } , sampleData.size() , 1<<Depth.value );


	// Initial solve up to the coarse depth
	GridSamples::TreeEstimator< Dim , 1 , false > estimator( DensityKernelRadius.value , KernelDepth.value , [&]( size_t i ){ return sampleData[i].first; } , orderedSampler , false , SamplesPerNode.value );
	if( Verbosity.value ) std::cout << pMeter( "Got density" ) << std::endl;

	std::function< std::pair< Point< double , Dim > , SampleData > ( size_t ) > SampleFunctor = [&]( size_t idx ){ return sampleData[idx]; };

	SymPR::Reconstructor< Dim , Sym , Energy > recon
	(
		hierarchicalIndexer ,
		Depth.value , MinDepth.value ,
		SampleFunctor , sampleData.size() ,
		orderedSampler ,
		estimator ,
		(int)AdaptiveDilationRadius.value ,
		FirstOrder.set ,
		BoundaryStiffnessWeight.value ,
		ScreeningWeight.value ,
		true ,
		TensorKernelRadius.value ,
		Verbosity.value
	);

	if( Verbosity.value ) std::cout << pMeter( "Initialized" ) << std::endl;
	
	std::vector<Eigen::VectorXd> solutions = recon.solve
	(
		SolveDepth.value ,
		hierarchicalIndexer ,
		solution , true ,
		false , IterationMultiplier.value , 
		CoarseIterations.value ,
		Iterations.value ,
#ifdef USE_NL_OPT
		GradientMemory.value ,
#endif // USE_NL_OPT
		PoissonSolvers::UpdateType::SERIAL_GAUSS_SEIDEL ,
		PoissonSolvers::UpdateType::PARALLEL_GAUSS_SEIDEL ,
		Verbosity.value
	);

	solution = solutions.back();

	if( Verbosity.value ) std::cout << pMeter( "Solved" ) << std::endl;

	if( !NoSign.set )
	{
		// Multiply by +/-1 so that the average is positive
		double sum = 0;
		for( unsigned int i=0 ; i<solution.size() ; i++ ) sum += solution[i];
		if( sum<0 ) solution *= -1;
	}

	if( Out.set )
	{
		XForm< double , Dim+1 > voxelToWorld;
		{
			SquareMatrix< double , Dim+1 > gridToCube = SquareMatrix< double , Dim+1 >::Identity();
			for( unsigned int d=0 ; d<Dim ; d++ ) gridToCube(d,d) = 1./(1<<ExtractDepth.value);
			voxelToWorld = unitCubeToWorld * gridToCube;
		}
		Miscellany::PerformanceMeter _pMeter( '.' );
		using Node = typename IsoTree< Dim , CoDim >::Node;

		auto CoefficientFunctor = [&]( unsigned int d , size_t f ){ return solutions[d][f]; };
		IsoTree< Dim , CoDim > isoTree( hierarchicalIndexer , CoefficientFunctor , ExtractDepth.value );
		if( Verbosity.value ) std::cout << _pMeter( "Iso-tree" ) << std::endl;

		isoTree.refineZeroCrossing( true );
		if( Verbosity.value ) std::cout << _pMeter( "Dilated" ) << std::endl;

		std::string ext = ToLower( GetFileExtension( Out.value ) );
		if( ext=="grid" )
		{
			Eigen::VectorXd _v = hierarchicalIndexer.regularCoefficients( solutions );
			RegularGrid< Dim , double > grid;
			grid.resize( ( 1<< ExtractDepth.value ) + 1 );
			{
				for( size_t i=0 ; i<grid.resolution() ; i++ ) grid[i] = _v[i];
				grid.write( Out.value , voxelToWorld );
			}
			if( Verbosity.value ) std::cout << pMeter( "Output grid" ) << std::endl;
		}
		else if( ext=="ply" )
		{
			_pMeter.reset();
			if constexpr( Dim==3 )
			{
				std::vector< Point< double , Dim > > vertices;
				std::vector< std::vector< unsigned int > > polygons;
				std::map< std::pair< typename RegularGrid< Dim >::Index , typename RegularGrid< Dim >::Index > , unsigned int > edgeToVertex;
				std::map< typename RegularGrid< Dim >::Index , double > cornerToValue;

				auto EdgeToVertex = [&]( typename RegularGrid< Dim >::Index C1 , typename RegularGrid< Dim >::Index C2 , Point< double , Dim > v )
					{
						if( C2<C1 ) std::swap( C1 , C2 );
						std::pair< typename RegularGrid< Dim >::Index , typename RegularGrid< Dim >::Index > key( C1 , C2 );
						auto iter = edgeToVertex.find( key );
						if( iter!=edgeToVertex.end() ) return iter->second;
						else
						{
							unsigned int sz = static_cast< unsigned int >( vertices.size() );
							vertices.push_back( v );
							edgeToVertex[key] = sz;
							return sz;
						}
					};

				std::vector< const typename IsoTree< Dim , CoDim >::Node * > levelSetNodes = isoTree.levelSetNodes( true );
				if( Verbosity.value ) std::cout << _pMeter( "Level-set nodes" ) << std::endl;

				auto LevelSetNodeIndex = [&]( size_t i )
					{
						Point< unsigned int , Dim > I;
						Hat::Index< Dim > _I = levelSetNodes[i]->offset();
						for( unsigned int d=0 ; d<Dim ; d++ ) I[d] = (unsigned int)_I[d];
						return I;
					};

				typename IsoSurface3D< double , unsigned int >::CellPolygonExtractor cellPolygonExtractor( true );
				double values[1<<Dim];
				for( size_t i=0 ; i<levelSetNodes.size() ; i++ )
				{
					typename RegularGrid< Dim >::Index I = LevelSetNodeIndex( i );
					for( unsigned int c=0 ; c<(1<<Dim) ; c++ )
					{
						typename RegularGrid< Dim >::Index Off;
						Cube::FactorCornerIndex( c , Off[0] , Off[1] , Off[2] );
						auto iter = cornerToValue.find( I+Off );
						if( iter!=cornerToValue.end() ) values[c] = iter->second;
						else values[c] = cornerToValue[I+Off] = isoTree.functionValue( I+Off )[0];
					}
					std::vector< std::vector< unsigned int > > _polygons = cellPolygonExtractor.extract( I , values , 0. , EdgeToVertex );
					for( unsigned int j=0 ; j<_polygons.size() ; j++ ) polygons.push_back( _polygons[j] );
				}
				if( Verbosity.value ) std::cout << _pMeter( "Level-set polygons" ) << std::endl;

				for( unsigned int i=0 ; i<vertices.size() ; i++ ) vertices[i] = voxelToWorld( vertices[i] );

				if( Triangulate.set )
				{
					std::vector< SimplexIndex< 2 > > triangles;
					for( size_t i=0 ; i<polygons.size() ; i++ )
					{
						std::vector< Point< double , Dim > > polygon( polygons[i].size() );
						for( unsigned int j=0 ; j<polygons[i].size() ; j++ ) polygon[j] = vertices[ polygons[i][j] ];
						std::vector< SimplexIndex< 2 > > _triangles;
						MinimalAreaTriangulation::GetTriangulation( polygon , _triangles );
						for( unsigned int j=0 ; j<_triangles.size() ; j++ )
						{
							for( unsigned int k=0 ; k<3 ; k++ ) _triangles[j][k] = polygons[i][ _triangles[j][k] ];
							triangles.push_back( _triangles[j] );
						}
					}
					using Factory = VertexFactory::PositionFactory< double , Dim >;
					Factory factory;
					PLY::WriteTriangles( Out.value , factory , vertices , triangles , PLY_BINARY_NATIVE );
				}
				else
				{
					using Factory = VertexFactory::PositionFactory< double , Dim >;
					Factory factory;
					PLY::WritePolygons( Out.value , factory , vertices , polygons , PLY_BINARY_NATIVE );
				}
			}
			else
			{
				MarchingSimplices::SimplicialMesh< Dim , unsigned int , Point< double , Dim > > sMesh;
				{
					_pMeter.reset();
					std::vector< const typename IsoTree< Dim , CoDim >::Node * > levelSetNodes = isoTree.levelSetNodes( true );
					if( Verbosity.value ) std::cout << _pMeter( "Level-set nodes" ) << std::endl;

					auto LevelSetNodeIndex = [&]( size_t i )
						{
							Point< unsigned int , Dim > I;
							Hat::Index< Dim > _I = levelSetNodes[i]->offset();
							for( unsigned int d=0 ; d<Dim ; d++ ) I[d] = (unsigned int)_I[d];
							return I;
						};
					sMesh = MarchingSimplices::RegularSubGridTriangulation< Dim >( levelSetNodes.size() , LevelSetNodeIndex , true , false );
				}
				if( Verbosity.value ) std::cout << pMeter( "Triangulation" ) << std::endl;

				std::vector< Point< double , CoDim > > values( sMesh.vertices.size() );
				{
					unsigned int res = 1<<ExtractDepth.value;
					auto SimplexElement = [&]( size_t i )
						{
							Point< double , Dim > p;
							for( unsigned int d=0 ; d<=Dim ; d++ ) p += sMesh.vertices[ sMesh.simplexIndices[i][d] ];
							p /= Dim+1;
							Hat::Index< Dim > E;
							for( unsigned int d=0 ; d<Dim ; d++ ) E[d] = std::max< int >( 0 , std::min< int >( res-1 , (int)floor( p[d] ) ) );
							return E;
						};
					std::vector< Hat::Index< Dim > > vertexElements( sMesh.vertices.size() );
					for( unsigned int s=0 ; s<sMesh.simplexIndices.size() ; s++ )
					{
						Hat::Index< Dim > E = SimplexElement(s);
						for( unsigned int d=0 ; d<=Dim ; d++ ) vertexElements[ sMesh.simplexIndices[s][d] ] = E;
					}
					ThreadPool::ParallelFor
					(
						0 , sMesh.vertices.size() ,
						[&]( unsigned int t , size_t v )
						{
							Point< double , Dim > p = sMesh.vertices[v];
							Hat::Index< Dim > F;
							for( unsigned int d=0 ; d<Dim ; d++ ) F[d] = (int)floor( p[d]+0.5 );
							values[v] = isoTree.functionValue( F );
						}
					);
				}
				if( Verbosity.value ) std::cout << _pMeter( "Sampled" ) << std::endl;

				MarchingSimplices::SimplicialMesh< Dim-CoDim , unsigned int , Point< double , Dim > > levelSet;
				levelSet = MarchingSimplices::LevelSet< double >( sMesh , [&]( size_t idx ){ return values[idx]; } , Point< double , CoDim >() );
				if( Verbosity.value ) std::cout << _pMeter( "Level set" ) << std::endl;

				for( unsigned int i=0 ; i<levelSet.vertices.size() ; i++ ) levelSet.vertices[i] = voxelToWorld( levelSet.vertices[i] );

				{
					using Factory = VertexFactory::PositionFactory< double , Dim >;
					Factory factory;
					PLY::WriteSimplices( Out.value , factory , levelSet.vertices , levelSet.simplexIndices , PLY_BINARY_NATIVE );
				}
			}
			if( Verbosity.value ) std::cout << pMeter( "Output mesh" ) << std::endl;
		}
		else MK_WARN_ONCE( "Unsupported file type: " , Out.value );
	}
}

template< unsigned int Dim >
std::vector< std::pair< Point< double , Dim > , Hat::SquareMatrix< double , Dim , Sym > > > ReadSamples( std::string fileName , bool verbose )
{
	Miscellany::PerformanceMeter pMeter( '.' );

	std::vector< Point< double , Dim > > positions = Samples::ReadPositions< double , Dim >( fileName );
	if( verbose )
	{
		std::stringstream ss;
		ss << "Read " << positions.size() << " points";
		std::cout << pMeter( ss.str() ) << std::endl;
	}

	std::vector< std::pair< Point< double , Dim > , Hat::SquareMatrix< double , Dim , Sym > > > sampleData( positions.size() );
	for( unsigned int i=0 ; i<positions.size() ; i++ ) sampleData[i].first = positions[i];

	{
		std::vector< SquareMatrix< double , Dim > > normalCovariances = UnsignedNormals::GetNormalCovariances( positions , NearestNeighbors.value , Verbosity.value>1 );
		for( unsigned int i=0 ; i<sampleData.size() ; i++ ) sampleData[i].second = normalCovariances[i];
		if( verbose ) std::cout << pMeter( "Got normals" ) << std::endl;
	}
	return sampleData;
}