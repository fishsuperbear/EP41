/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef CBUFFER_HPP
#define CBUFFER_HPP

#include <atomic>
#include <cstdint>
#include <vector>
#include <mutex>

#include "CBufferPool.hpp"
//#include "CUtils.hpp"
#include "hw_nvmedia_eventhandler_common_impl.h"
#include "nvscisync.h"

class CBuffer
{
public:
    static const uint32_t MAX_PRENVSCISYNCFENCES = 16U;

    void Init( CBufferPool<CBuffer>* pBufferPool )
    {
        m_pBufferPool = pBufferPool;
    }

    virtual ~CBuffer()
    {
        const uint32_t refcount = m_uReferenceCount.load();
        if ( refcount != 0U )
        {
            LOG_WARN( "Destroying buffer with reference count: %u\n", refcount );
        }
    }

    SIPLStatus Deinit()
    {
        try
        {
            unique_lock<mutex> postFenceLock( m_postFenceMutex );
            if ( m_hasPostFence )
            {
                LOG_WARN( "Clearing buffer postfence on deinit\n" );
                NvSciSyncFenceClear( &m_postFence );
                m_hasPostFence = false;
            }
            postFenceLock.unlock();

            unique_lock<mutex> preFenceLock( m_preFenceMutex );
            for ( uint32_t idx = 0U; idx < m_preFenceCount; idx++ )
            {
                NvSciSyncFenceClear( &m_preFences[idx] );
            }
            m_preFenceCount = 0U;
            preFenceLock.unlock();

            return NVSIPL_STATUS_OK;
        }
        catch ( exception& e )
        {
            return HandleException( &e );
        }
    }

    virtual void AddRef()
    {
        ++m_uReferenceCount;
    }

    virtual SIPLStatus Release()
    {
        try
        {
            uint32_t refcount = m_uReferenceCount.load();
            while ( true )
            {
                if ( refcount == 0U )
                {
                    LOG_ERR( "Attempt to double release buffer\n" );
                    return NVSIPL_STATUS_BAD_ARGUMENT;
                }

                const bool success = m_uReferenceCount.compare_exchange_strong( refcount, refcount - 1U );
                if ( success )
                {
                    refcount--;
                    if ( refcount == 0U )
                    {
                        unique_lock<mutex> postFenceLock( m_postFenceMutex );
                        NvSciSyncFenceClear( &m_postFence );
                        m_hasPostFence = false;
                        postFenceLock.unlock();

                        const SIPLStatus status = m_pBufferPool->Add( this );
                        if ( status != NVSIPL_STATUS_OK )
                        {
                            LOG_ERR( "Buffer pool add failed\n" );
                            return status;
                        }
                    }
                    return NVSIPL_STATUS_OK;
                }
            }
        }
        catch ( exception& e )
        {
            return HandleException( &e );
        }
    }

    virtual NvSciBufObj GetNvSciBufObj() const
    {
        return m_pSciBuf;
    }

    virtual SIPLStatus AddNvSciSyncPrefence( const NvSciSyncFence& prefence )
    {
        try
        {
            unique_lock<mutex> preFenceLock( m_preFenceMutex );
            if ( m_preFenceCount < MAX_PRENVSCISYNCFENCES )
            {
                const NvSciError sciErr = NvSciSyncFenceDup( &prefence, &m_preFences[m_preFenceCount] );
                if ( sciErr != NvSciError_Success )
                {
                    LOG_ERR( "Failed to duplicate NvSci prefence for add" );
                    return NVSIPL_STATUS_ERROR;
                }
                m_preFenceCount++;
            }
            else
            {
                LOG_ERR( "Too many NvSci prefences being added. max:", MAX_PRENVSCISYNCFENCES );
                return NVSIPL_STATUS_ERROR;
            }
            preFenceLock.unlock();
        }
        catch( exception& e )
        {
            return HandleException( &e );
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus GetEOFNvSciSyncFence( NvSciSyncFence* postfence )
    {
        try
        {
            unique_lock<mutex> postFenceLock( m_postFenceMutex );
            if ( !m_hasPostFence )
            {
                LOG_ERR( "Retrieving non-existent NvSci postfence" );
                return NVSIPL_STATUS_ERROR;
            }

            const NvSciError sciErr = NvSciSyncFenceDup( &m_postFence, postfence );
            if ( sciErr != NvSciError_Success )
            {
                LOG_ERR( "Failed to duplicate NvSci postfence for get" );
                return NVSIPL_STATUS_ERROR;
            }
            postFenceLock.unlock();

            return NVSIPL_STATUS_OK;
        }
        catch( exception& e )
        {
            return HandleException( &e );
        }
    }

    virtual SIPLStatus GetNvSciSyncPrefences( NvSciSyncFence prefences[], uint32_t* num_fence_inout )
    {
        try
        {
            unique_lock<mutex> preFenceLock( m_preFenceMutex );
            if ( *num_fence_inout >= m_preFenceCount )
            {
                for ( uint32_t idx = 0U; idx < m_preFenceCount; idx++ )
                {
                    const NvSciError sciErr = NvSciSyncFenceDup( &m_preFences[idx], &prefences[idx] );
                    if ( sciErr != NvSciError_Success )
                    {
                        LOG_ERR( "Failed to duplicate NvSci prefence for get" );
                        return NVSIPL_STATUS_ERROR;
                    }
                }
                *num_fence_inout = m_preFenceCount;
            }
            else
            {
                LOG_ERR( "Prefence array too small to copy all fences (given < needed)" );
                return NVSIPL_STATUS_ERROR;
            }
            preFenceLock.unlock();
        }
        catch( exception& e )
        {
            return HandleException( &e );
        }

        return NVSIPL_STATUS_OK;
    }

    virtual SIPLStatus ClearNvSciSyncPrefences()
    {
        try
        {
            unique_lock<mutex> preFenceLock( m_preFenceMutex );
            for ( uint32_t idx = 0U; idx < m_preFenceCount; idx++ )
            {
                NvSciSyncFenceClear( &m_preFences[idx] );
            }
            m_preFenceCount = 0U;
            preFenceLock.unlock();

            return NVSIPL_STATUS_OK;
        }
        catch( exception& e )
        {
            return HandleException( &e );
        }
    }

    virtual SIPLStatus UpdateEOFNvSciSyncFence( const NvSciSyncFence& postfence )
    {
        try
        {
            unique_lock<mutex> postFenceLock( m_postFenceMutex );
            const NvSciError sciErr = NvSciSyncFenceDup( &postfence, &m_postFence );
            if ( sciErr != NvSciError_Success )
            {
                LOG_ERR( "Failed to duplicate NvSci postfence for update" );
                return NVSIPL_STATUS_ERROR;
            }
            m_hasPostFence = true;
            postFenceLock.unlock();

            return NVSIPL_STATUS_OK;
        }
        catch( exception& e )
        {
            return HandleException( &e );
        }
    }

    NvSciSyncFence m_postFence = NvSciSyncFenceInitializer;
    bool           m_hasPostFence {};
    mutex          m_postFenceMutex;

    NvSciSyncFence m_preFences[MAX_PRENVSCISYNCFENCES] = { NvSciSyncFenceInitializer };
    uint32_t       m_preFenceCount {};
    mutex          m_preFenceMutex;

    virtual void SetNvSciBufObj( NvSciBufObj sciBufObj )
    {
        m_pSciBuf = sciBufObj;
    }

private:
    NvSciBufObj                 m_pSciBuf {};
    CBufferPool<CBuffer>*       m_pBufferPool {nullptr};
    atomic<uint32_t>            m_uReferenceCount {};
};

#endif // CBUFFER_HPP
