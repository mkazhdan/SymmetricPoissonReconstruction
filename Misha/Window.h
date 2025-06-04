/*
Copyright (c) 2016, Michael Kazhdan
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

#ifndef WINDOW_INCLUDED
#define WINDOW_INCLUDED

#include <functional>
#include "Array.h"
#include "ParameterPack.h"
#include "MultiThreading.h"

namespace MishaK
{
	namespace Window
	{
		template< unsigned int Res , unsigned int ... Ress >
		constexpr unsigned int Size( void )
		{
			if constexpr( sizeof...(Ress)==0 ) return Res;
			else                               return Res * Size< Ress... >();
		}

		template< unsigned int Dim , unsigned int Res >
		constexpr unsigned int IsotropicSize( void )
		{
			if constexpr( Dim==1 ) return Res;
			else                   return Res * IsotropicSize< Dim-1 , Res >();
		}

		template< unsigned int Res , unsigned int ... Ress >
		struct Index
		{
			template< unsigned int Idx , unsigned int ... Idxs >
			static constexpr unsigned int I( void )
			{
				static_assert( sizeof...(Ress)==sizeof...(Idxs) , "[ERROR] sizes don't match" );
				if constexpr( sizeof...(Ress)==0 ) return Idx;
				else                               return Idx * Size< Ress ... >() + Index< Ress... >::template I< Idxs... >();
			}
		};

		template< unsigned int Dim , unsigned int Res >
		struct IsotropicIndex
		{
			template< unsigned int Idx >
			static constexpr unsigned int I( void )
			{
				if constexpr( Dim==1 ) return Idx;
				else                   return Idx * IsotropicSize< Dim-1 , Res >() + IsotropicIndex< Dim-1 , Res >::template I< Idx >();
			}
		};

		template< unsigned int Res , unsigned int ... Ress >
		unsigned int GetIndex( const unsigned int idx[] )
		{
			if constexpr( sizeof...(Ress)==0 ) return idx[0];
			else                               return idx[0] * Size< Ress ... >() + GetIndex< Ress... >( idx+1 );
		}

		template< unsigned int Res , unsigned int ... Ress >
		unsigned int GetIndex( const int idx[] )
		{
			if constexpr( sizeof...(Ress)==0 ) return idx[0];
			else                               return idx[0] * Size< Ress ... >() + GetIndex< Ress... >( idx+1 );
		};

		template< unsigned int Dim , unsigned int Res >
		unsigned int IsotropicGetIndex( const unsigned int idx[] )
		{
			if constexpr( Dim==1 ) return idx[0];
			else                   return idx[0] * IsotropicSize< Dim-1 , Res >() + IsotropicGetIndex< Dim-1 , Res >( idx+1 );
		}

		template< unsigned int Dim , unsigned int Res >
		unsigned int IsotropicGetIndex( const int idx[] )
		{
			if constexpr( Dim==1 ) return idx[0];
			else                   return idx[0] * IsotropicSize< Dim-1 , Res >() + IsotropicGetIndex< Dim-1 , Res >( idx+1 );
		};

		template< class Data , unsigned int ... Res > struct ConstSlice{};

		template< class Data , unsigned int Res , unsigned int ... Ress >
		struct ConstSlice< Data , Res , Ress... >
		{
			using data_type = Data;
			using data_reference_type = const Data &;
			using const_data_reference_type = const Data &;
			static constexpr unsigned int Size( void ){ return Window::Size< Res , Ress... >(); }

			ConstSlice(      Pointer( Data ) d ) : data(d) {}
			ConstSlice( ConstPointer( Data ) d ) : data(d) {}

			std::conditional_t< sizeof...(Ress)==0 , data_reference_type , ConstSlice< Data , Ress... > > operator[]( int idx ) const
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               return ConstSlice< Data , Ress... >( data + Window::Size< Ress... >() * idx );
			}
			data_reference_type operator()( const          int idx[] ) const { return data[ GetIndex< Res , Ress... >( idx ) ]; }
			data_reference_type operator()( const unsigned int idx[] ) const { return data[ GetIndex< Res , Ress... >( idx ) ]; }
			ConstPointer( Data ) data;
		};

		template< typename Data , unsigned Dim , unsigned int Res , unsigned int ... Ress >
		struct _IsotropicConstSlice{ using Type = std::conditional_t< Dim==1 , ConstSlice< Data , Res , Ress... > , _IsotropicConstSlice< Data , Dim-1 , Res , Res , Ress... > >; };
		template< typename Data , unsigned int Dim , unsigned int Res >
		using IsotropicConstSlice = typename _IsotropicConstSlice< Data , Dim , Res >::Type;


		template< class Data , unsigned int ... Res > struct Slice{};

		template< class Data , unsigned int Res , unsigned int ... Ress >
		struct Slice< Data , Res , Ress... >
		{
			using data_type = Data;
			using data_reference_type = Data &;
			using const_data_reference_type = const Data &;
			static constexpr unsigned int Size( void ){ return Window::Size< Res , Ress... >(); }

			Slice( Pointer( Data ) d ) : data(d) {}
			std::conditional_t< sizeof...(Ress)==0 , data_reference_type , Slice< Data , Ress... > > operator[]( int idx )
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               return Slice< Data , Ress... >( data + Window::Size< Ress... >() * idx );
			}
			std::conditional_t< sizeof...(Ress)==0 , const_data_reference_type , ConstSlice< Data , Ress... > > operator[]( int idx ) const
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               return ConstSlice< Data , Ress... >( data + Window::Size< Ress... >() * idx );
			}
			data_reference_type operator()( const int idx[] )
			{
				if constexpr( sizeof...(Ress)==0 ) return operator[]( idx[0] );
				else                               return operator[]( idx[0] )( idx+1 );
			}
			const_data_reference_type operator()( const int idx[] ) const
			{
				if constexpr( sizeof...(Ress)==0 ) return operator[]( idx[0] );
				else                               return operator[]( idx[0] )( idx+1 );
			}
			operator ConstSlice< Data , Res , Ress... >() const { return ConstSlice< Data , Res , Ress... >( ( ConstPointer( Data ) )data ); }
			Pointer( Data ) data;
		};

		template< typename Data , unsigned Dim , unsigned int Res , unsigned int ... Ress >
		struct _IsotropicSlice{ using Type = std::conditional_t< Dim==1 , Slice< Data , Res , Ress... > , _IsotropicSlice< Data , Dim-1 , Res , Res , Ress... > >; };
		template< typename Data , unsigned int Dim , unsigned int Res >
		using IsotropicSlice = typename _IsotropicSlice< Data , Dim , Res >::Type;


		template< class Data , unsigned int Res , unsigned int ... Ress >
		struct StaticWindow
		{
			using const_window_slice_type = ConstSlice< Data , Res , Ress... >;
			using window_slice_type = Slice< Data , Res , Ress... >;
			using data_type = Data ;
			static constexpr unsigned int Size( void ){ return Window::Size< Res , Ress... >(); }

			std::conditional_t< sizeof...(Ress)==0 , Data & , Slice< Data , Ress... > > operator[]( int idx )
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               return Slice< Data , Ress... >( GetPointer( data , Size() ) + Window::Size< Ress... >() * idx );
			}

			std::conditional_t< sizeof...(Ress)==0 , const Data & , ConstSlice< Data , Ress... > > operator[]( int idx ) const
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               return ConstSlice< Data , Ress... >( ( ConstPointer( Data ) )GetPointer( data , Size() ) + Window::Size< Ress... >() * idx );
			}

			Slice< Data , Res , Ress... > operator()( void ){ return Slice< Data , Res , Ress... >( GetPointer( data , Size() ) ); }

			ConstSlice< Data , Res , Ress... > operator()( void ) const { return ConstSlice< Data , Res , Ress... >( ( ConstPointer( Data ) )GetPointer( data , Size() ) ); }

			Data& operator()( const unsigned int idx[] ){ return (*this)()( idx ); }
			Data& operator()( const          int idx[] ){ return (*this)()( idx ); }

			const Data& operator()( const unsigned int idx[] ) const { return data[ GetIndex< Res , Ress... >( idx ) ]; }
			const Data& operator()( const          int idx[] ) const { return data[ GetIndex< Res , Ress... >( idx ) ]; }

			Data data[ Window::Size< Res , Ress... >() ];
		};

		template< typename Data , unsigned Dim , unsigned int Res , unsigned int ... Ress >
		struct _IsotropicStaticWindow{ using Type = typename _IsotropicStaticWindow< Data , Dim-1 , Res , Res , Ress... >::Type; };

		template< typename Data , unsigned int Res , unsigned int ... Ress >
		struct _IsotropicStaticWindow< Data , 1 , Res , Ress... >{ using Type = StaticWindow< Data , Res , Ress... >; };

		template< typename Data , unsigned int Dim , unsigned int Res >
		using IsotropicStaticWindow = typename _IsotropicStaticWindow< Data , Dim , Res >::Type;


		template< class Data , unsigned int Res , unsigned int ... Ress >
		struct DynamicWindow
		{
			using const_window_slice_type = ConstSlice< Data , Res , Ress... >;
			using window_slice_type = Slice< Data , Res , Ress... >;
			using data_type = Data;
			static constexpr unsigned int Size( void ){ return Window::Size< Res , Ress... >(); }

			std::conditional_t< sizeof...(Ress)==0 , Data & , Slice< Data , Ress... > > operator[]( int idx )
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               return Slice< Data , Ress... >( data + Window::Size< Ress... >() * idx );
			}

			std::conditional_t< sizeof...(Ress)==0 , const Data & , ConstSlice< Data , Ress... > > operator[]( int idx ) const
			{
				if constexpr( sizeof...(Ress)==0 ) return data[idx];
				else                               ConstSlice< Data , Ress... >( ( ConstPointer( Data ) )( data + Window::Size< Ress... >() * idx ) );
			}

			Slice< Data , Res , Ress... > operator()( void ){ return Slice< Data , Res , Ress... >( data ); }
			ConstSlice< Data , Res , Ress... > operator()( void ) const { return ConstSlice< Data , Res , Ress... >( ( ConstPointer( Data ) )data ); }

			Data& operator()( const int idx[] ){ return (*this)()( idx ); }
			const Data& operator()( const int idx[] ) const { return (*this)()( idx ); }

			DynamicWindow( void ){ data = NewPointer< Data >( Size() ); }

			~DynamicWindow( void ){ DeletePointer( data ); }

			Pointer( Data ) data;
		};

		template< typename Data , unsigned Dim , unsigned int Res , unsigned int ... Ress >
		struct _IsotropicDynamicWindow{ using Type = typename _IsotropicDynamicWindow< Data , Dim-1 , Res , Res , Ress... >::Type; };

		template< typename Data , unsigned int Res , unsigned int ... Ress >
		struct _IsotropicDynamicWindow< Data , 1 , Res , Ress... >{ using Type = DynamicWindow< Data , Res , Ress... >; };

		template< typename Data , unsigned int Dim , unsigned int Res >
		using IsotropicDynamicWindow = typename _IsotropicDynamicWindow< Data , Dim , Res >::Type;

		// Recursive loop iterations for processing window slices
		//		WindowDimension: the the window slice
		//		IterationDimensions: the number of dimensions to process
		//		Res: the resolution of the window

		template< unsigned int WindowDimension , unsigned int IterationDimensions , unsigned int CurrentIteration > struct _Loop;

		template< unsigned int WindowDimension , unsigned int IterationDimensions=WindowDimension >
		struct Loop
		{
			template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
			static void Run( int begin , int end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
			{
				_Loop< WindowDimension , IterationDimensions , IterationDimensions >::Run( begin , end , updateState , function , w ... ); 
			}
			template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
			static void Run( const int* begin , const int* end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
			{
				_Loop< WindowDimension , IterationDimensions , IterationDimensions >::Run( begin , end , updateState , function , w ... ); 
			}
			template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
			static void Run( ParameterPack::UIntPack< Begin ... > begin , ParameterPack::UIntPack< End ... > end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
			{
				_Loop< WindowDimension , IterationDimensions , IterationDimensions >::Run( begin , end , updateState , function , w ... ); 
			}

			template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
			static void RunParallel( int begin , int end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
			{
				_Loop< WindowDimension , IterationDimensions , IterationDimensions >::RunParallel( begin , end , updateState , function , w ... ); 
			}
			template< typename UpdateFunction , typename ProcessFunction , class ... Windows >
			static void RunParallel( const int* begin , const int* end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
			{
				_Loop< WindowDimension , IterationDimensions , IterationDimensions >::RunParallel( begin , end , updateState , function , w ... ); 
			}
			template< unsigned int ... Begin , unsigned int ... End , typename UpdateFunction , typename ProcessFunction , class ... Windows >
			static void RunParallel( ParameterPack::UIntPack< Begin ... > begin , ParameterPack::UIntPack< End ... > end , UpdateFunction updateState , ProcessFunction function , Windows ... w )
			{
				_Loop< WindowDimension , IterationDimensions , IterationDimensions >::RunParallel( begin , end , updateState , function , w ... ); 
			}
		};

#include "Window.inl"

	}
}
#endif // WINDOW_INCLUDED
