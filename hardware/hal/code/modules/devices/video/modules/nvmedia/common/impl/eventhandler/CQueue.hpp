/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CQUEUE_HPP
#define CQUEUE_HPP

#include <mutex>
#include <cstdint>
#include <vector>
#include <exception>
//#include "CUtils.hpp"

using namespace std;
/**
 * A templatized Queue class. This class implements a non-blocking thread-safe queue.
 */
template <class T>
class CQueue
{
public:
    SIPLStatus Init( const size_t uSize )
    {
        SIPLStatus status = NVSIPL_STATUS_OK;
        if ( uSize == 0U )
        {
            LOG_ERR( "Invalid size for queue\n" );
            status = NVSIPL_STATUS_BAD_ARGUMENT;
            goto FUNC_END;
        }

        try
        {
            m_vBuffer.reserve( uSize );
        }
        catch ( exception& e )
        {
            status = HandleException( &e );
        }
        m_uSize = uSize;
FUNC_END:
        return status;
    }

    SIPLStatus Add( const T& item )
    {
        SIPLStatus status = NVSIPL_STATUS_OK;

        try
        {
            unique_lock<mutex> const lock( m_oMutex );
            if ( m_uCount >= m_uSize )
            {
                LOG_ERR( "Queue is full\n" );
                status = NVSIPL_STATUS_ERROR;
                goto FUNC_END;
            }

            m_vBuffer[m_uWriteIndex] = item;
            m_uWriteIndex = ( m_uWriteIndex + 1U ) % m_uSize;
            m_uCount++;
        }
        catch ( exception&e )
        {
            status = HandleException( &e );
        }
FUNC_END:
        return status;
    }

    SIPLStatus Get( T& item )
    {
        SIPLStatus status = NVSIPL_STATUS_OK;

        try {
            unique_lock<mutex> const lock( m_oMutex );
            if ( m_uCount == 0U )
            {
                LOG_ERR( "Queue is empty\n" );
                status = NVSIPL_STATUS_ERROR;
                goto FUNC_END;
            }

            item = m_vBuffer[m_uReadIndex];
            m_uReadIndex = ( m_uReadIndex + 1U ) % m_uSize;
            m_uCount--;
        }
        catch ( exception& e )
        {
            status = HandleException( &e );
        }
FUNC_END:
        return status;
    }

    bool IsEmpty() const
    {
        return ( m_uCount == 0U );
    }

    size_t GetCount() const
    {
        return m_uCount;
    }

private:
    vector<T> m_vBuffer {};
    mutex     m_oMutex;
    size_t    m_uSize {};
    size_t    m_uReadIndex {};
    size_t    m_uWriteIndex {};
    size_t    m_uCount {};
};

#endif // CQUEUE_HPP
