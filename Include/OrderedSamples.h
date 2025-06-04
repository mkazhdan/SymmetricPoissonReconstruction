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

#ifndef ORDERED_SAMPLES_INCLUDED
#define ORDERED_SAMPLES_INCLUDED

#include <vector>
#include "Misha/Geometry.h"
#include "Misha/RegularGrid.h"
#include "Hat.h"


namespace MishaK
{
	template< unsigned int Dim , typename ... > struct OrderedSampleType;
	template< unsigned int Dim > struct OrderedSampleType< Dim >{ using Type = Point< double , Dim >; };
	template< unsigned int Dim , typename Data > struct OrderedSampleType< Dim , Data >{ using Type = std::pair< Point< double , Dim > , Data >; };

	template< unsigned int Dim >
	struct OrderedSampler
	{
		template< typename SampleFunctor /* = std::function< Point< double , Dim > ( size_t ) > */ >
		OrderedSampler( SampleFunctor && SampleF , size_t sampleNum , unsigned int res );
		size_t size( void ) const;
		size_t numSamples( void ) const;
		const std::pair< typename RegularGrid< Dim >::Index , std::vector< size_t > > &operator[]( size_t g ) const;
		unsigned int res( void ) const;
	protected:
		std::vector< std::pair< typename RegularGrid< Dim >::Index , std::vector< size_t > > > _sampleIndices;
		unsigned int _res;
	};

	template< unsigned int Dim , typename ... Data >
	struct OrderedSamples
	{
		struct _Sample;
		static_assert( sizeof...(Data)==0 || sizeof...(Data)==1 , "[ERROR] Multiple data not expected" );
		using Sample = typename OrderedSampleType< Dim , Data... >::Type;

		template< typename SampleFunctor /* = std::function< Sample ( size_t ) > */ >
		OrderedSamples( SampleFunctor && SampleF , size_t sampleNum , unsigned int res );
		size_t size( void ) const;
		size_t numSamples( void ) const;
		const std::pair< typename RegularGrid< Dim >::Index , std::vector< Sample > > &operator[]( size_t g ) const;
		unsigned int res( void ) const;
	protected:
		std::vector< std::pair< typename RegularGrid< Dim >::Index , std::vector<Sample > > > _samples;
		unsigned int _res;
	};
#include "OrderedSamples.inl"
}

#endif // ORDERED_SAMPLES_INCLUDED