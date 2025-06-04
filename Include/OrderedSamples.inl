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

////////////////////
// OrderedSampler //
////////////////////
template< unsigned int Dim >
template< typename SampleFunctor /* = std::function< Point< double , Dim > ( size_t ) > */ >
OrderedSampler< Dim >::OrderedSampler( SampleFunctor && F , size_t sampleNum , unsigned int res )
	: _res(res)
{
	static_assert( std::is_convertible_v< SampleFunctor , std::function< Point< double , Dim > (size_t) > > , "[ERROR] Poorly form SampleFunctor" );



	struct IndexedSample
	{
		size_t idx;
		typename RegularGrid< Dim >::Index I;
		IndexedSample( size_t i , Point< double , Dim > p , unsigned int res ) : idx(i)
		{
			auto Index = [res]( Point< double , Dim > p )
				{
					typename RegularGrid< Dim >::Index I;
					for( unsigned int d=0 ; d<Dim ; d++ ) I[d] = std::max< int >( 0 , std::min< int >( (int)floor( p[d] * res ) , res-1 ) );
					return I;
				};
			I = Index( p );
		}
		bool operator < ( const IndexedSample & s ) const { return I<s.I; }
	};

	std::vector< IndexedSample > samples;
	samples.reserve( sampleNum );
	for( size_t i=0 ; i<sampleNum ; i++ ) samples.emplace_back( i , F(i) , res );
	std::sort( samples.begin() , samples.end() );
	{
		size_t count = 0 , start=0;
		while( start<samples.size() )
		{
			// compute the [start,end) range for samples mapping to the same cell
			size_t end;
			for( end=start ; end<samples.size() && samples[end].I==samples[start].I ; end++ );
			count++;
			start = end;
		}
		_sampleIndices.resize( count );
	}
	{
		size_t count = 0 , start=0;
		while( start<samples.size() )
		{
			// compute the [start,end) range for samples mapping to the same cell
			size_t end;
			for( end=start ; end<samples.size() && samples[end].I==samples[start].I ; end++ );
			_sampleIndices[count].first = samples[start].I;
			_sampleIndices[count].second.resize( end-start );
			for( size_t i=start ; i<end ; i++ ) _sampleIndices[count].second[i-start] = samples[i].idx;
			count++;
			start = end;
		}
	}
}

template< unsigned int Dim >
size_t OrderedSampler< Dim >::size( void ) const { return _sampleIndices.size(); }

template< unsigned int Dim >
size_t OrderedSampler< Dim >::numSamples( void ) const
{
	size_t count = 0;
	for( size_t i=0 ; i<_sampleIndices.size() ; i++ ) count += _sampleIndices[i].second.size();
	return count;
}

template< unsigned int Dim >
const std::pair< typename RegularGrid< Dim >::Index , std::vector< size_t > > &OrderedSampler< Dim >::operator[]( size_t g ) const { return _sampleIndices[g]; }

template< unsigned int Dim >
unsigned int OrderedSampler< Dim >::res( void ) const { return _res; }

////////////////////
// OrderedSamples //
////////////////////
template< unsigned int Dim , typename ... Data >
template< typename SampleFunctor /* = std::function< Sample ( size_t ) > */ >
OrderedSamples< Dim , Data... >::OrderedSamples( SampleFunctor && F , size_t sampleNum , unsigned int res )
	: _res(res)
{
	static_assert( std::is_convertible_v< SampleFunctor , std::function< Sample (size_t) > > , "[ERROR] Poorly form SampleFunctor" );

	struct IndexedSample
	{
		Sample sample;
		typename RegularGrid< Dim >::Index I;
		IndexedSample( Sample s , unsigned int res ) : sample(s)
		{
			auto Index = [res]( Point< double , Dim > p )
				{
					typename RegularGrid< Dim >::Index I;
					for( unsigned int d=0 ; d<Dim ; d++ ) I[d] = std::max< int >( 0 , std::min< int >( (int)floor( p[d] * res ) , res-1 ) );
					return I;
				};

			if constexpr( sizeof...(Data)==0 ) I = Index( s );
			else                               I = Index( s.first );
		}
		bool operator < ( const IndexedSample & s ) const { return I<s.I; }
	};

	std::vector< IndexedSample > samples;
	samples.reserve( sampleNum );
	for( size_t i=0 ; i<sampleNum ; i++ ) samples.emplace_back( F(i) , res );
	std::sort( samples.begin() , samples.end() );
	{
		size_t count = 0 , start=0;
		while( start<samples.size() )
		{
			// compute the [start,end) range for samples mapping to the same cell
			size_t end;
			for( end=start ; end<samples.size() && samples[end].I==samples[start].I ; end++ );
			count++;
			start = end;
		}
		_samples.resize( count );
	}
	{
		size_t count = 0 , start=0;
		while( start<samples.size() )
		{
			// compute the [start,end) range for samples mapping to the same cell
			size_t end;
			for( end=start ; end<samples.size() && samples[end].I==samples[start].I ; end++ );
			_samples[count].first = samples[start].I;
			_samples[count].second.resize( end-start );
			for( size_t i=start ; i<end ; i++ ) _samples[count].second[i-start] = samples[i].sample;
			count++;
			start = end;
		}
	}
}

template< unsigned int Dim , typename ... Data >
size_t OrderedSamples< Dim , Data... >::size( void ) const { return _samples.size(); }

template< unsigned int Dim , typename ... Data >
size_t OrderedSamples< Dim , Data... >::numSamples( void ) const
{
	size_t count = 0;
	for( size_t i=0 ; i<_samples.size() ; i++ ) count += _samples[i].second.size();
	return count;
}

template< unsigned int Dim , typename ... Data >
const std::pair< typename RegularGrid< Dim >::Index , std::vector< typename OrderedSamples< Dim , Data... >::Sample > > &OrderedSamples< Dim , Data... >::operator[]( size_t g ) const { return _samples[g]; }

template< unsigned int Dim , typename ... Data >
unsigned int OrderedSamples< Dim , Data... >::res( void ) const { return _res; }
