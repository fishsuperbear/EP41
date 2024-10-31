/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CBUFFERPOOL_HPP
#define CBUFFERPOOL_HPP

#include <vector>
#include <cstdint>
#include <memory>
#include <exception>
#include "CQueue.hpp"
//#include "CUtils.hpp"
#include "hw_nvmedia_eventhandler_common_impl.h"

using namespace std;

/**
 * @brief IBufferPoolCallback defines a callback interface for the
 * CBufferPool.
 */
class IBufferPoolCallback
{
public:
    /**
     * @brief When invoked, this method indicates a buffer has been
     * released and is now available in the pool.
     */
    virtual SIPLStatus OnBufferRelease() = 0;
};

template<class B>
class CBufferPool
{
    friend B;

public:

    SIPLStatus Deinit()
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        for ( unique_ptr<B>& buffer : m_vBuffers )
        {
            status = buffer->Deinit();
            if ( status != NVSIPL_STATUS_OK )
            {
               LOG_ERR( "Buffer deinitialization failed\n" );
               goto FUNC_END;
            }
        }
FUNC_END:
        return status;
   }

    SIPLStatus Init( const size_t uCount )
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        LOG_DBG( "Creating buffer pool with %u buffers\n", uCount );

        if ( uCount == 0U )
        {
            LOG_ERR( "Invalid size for buffer pool\n" );
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            goto FUNC_END;
        }

        try
        {
            // Allocate buffers & populate vector
            m_vBuffers.reserve( uCount );
            for ( size_t count = 0U; count != uCount; count++ )
            {
                unique_ptr<B> buffer( new B() );
                m_vBuffers.push_back( move( buffer ) );
            }

            status = m_oQueue.Init( uCount );
            if ( status != NVSIPL_STATUS_OK )
            {
                LOG_ERR( "Queue init failed\n" );
                goto FUNC_END;
            }

            /* Initialize buffer & add it to the queue */
            for ( unique_ptr<B>& buffer : m_vBuffers )
            {
                buffer->Init( this );
                status = m_oQueue.Add( buffer.get() );
                if ( status != NVSIPL_STATUS_OK )
                {
                    LOG_ERR( "Queue add failed\n" );
                    goto FUNC_END;
                }
            }
        }
        catch ( exception& e )
        {
            status = HandleException( &e );
        }
FUNC_END:
        return status;
    }

    SIPLStatus Get( B*& pBuffer )
    {
        SIPLStatus const status = m_oQueue.Get( pBuffer );
        if ( status != NVSIPL_STATUS_OK )
        {
            LOG_ERR( "Queue get failed\n" );
            goto FUNC_END;
        }

        pBuffer->AddRef();
FUNC_END:
        return status;
    }

    bool IsQueueEmpty() const
    {
        return m_oQueue.IsEmpty();
    }

    void SetCallback( IBufferPoolCallback* const pCallback )
    {
        m_pCallback = pCallback;
    }

    uint32_t GetCount() const
    {
        return m_oQueue.GetCount();
    }

private:

    SIPLStatus Add( B* const pBuffer )
    {
        SIPLStatus status = m_oQueue.Add( pBuffer );
        if ( status != NVSIPL_STATUS_OK )
        {
            LOG_ERR( "Queue add failed\n" );
            goto FUNC_END;
        }

        if ( m_pCallback != nullptr )
        {
            status = m_pCallback->OnBufferRelease();
            if ( status != NVSIPL_STATUS_OK )
            {
                LOG_ERR( "OnBufferRelease failed\n" );
                goto FUNC_END;
            }
        }
FUNC_END:
        return status;
    }

    CQueue<B*>             m_oQueue;
    vector<unique_ptr<B>>  m_vBuffers {};
    IBufferPoolCallback*   m_pCallback {nullptr};
};

#endif // CBUFFERPOOL_HPP
