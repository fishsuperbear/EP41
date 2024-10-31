/**
 * * --------------------------------------------------------------------
 * * |                                                                  |
 * * |     _         _    _ _______ ____         _____ ____  __  __     |
 * * |    (_)   /\  | |  | |__   __/ __ \       / ____/ __ \|  \/  |    |
 * * |     _   /  \ | |  | |  | | | |  | |     | |   | |  | | \  / |    |
 * * |    | | / /\ \| |  | |  | | | |  | |     | |   | |  | | |\/| |    |
 * * |    | |/ ____ \ |__| |  | | | |__| |  _  | |___| |__| | |  | |    |
 * * |    |_/_/    \_\____/   |_|  \____/  (_)  \_____\____/|_|  |_|    |
 * * |                                                                  |
 * * --------------------------------------------------------------------
 *
 *  * Copyright @ 2020 iAuto (Shanghai) Co., Ltd.
 *  * All Rights Reserved.
 *  *
 *  * Redistribution and use in source and binary forms, with or without
 *  * modification, are NOT permitted except as agreed by
 *  * iAuto (Shanghai) Co., Ltd.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS,
 *  * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 *
 * @file NCRWLock.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCRWLOCK_H_
#define INCLUDE_NCORE_NCRWLOCK_H_

#include <pthread.h>
#include <stdint.h>
#include <sys/types.h>

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
class NCRWLock {
   public:
    enum { PRIVATE = 0, SHARED = 1 };

    inline NCRWLock() { pthread_rwlock_init( &mRWLock, NULL ); }

    inline NCRWLock( __attribute__( ( unused ) ) const char *name ) {
        pthread_rwlock_init( &mRWLock, NULL );
    }

    inline NCRWLock( int type, __attribute__( ( unused ) ) const char *name = NULL ) {
        if ( type == SHARED ) {
            pthread_rwlockattr_t attr;
            pthread_rwlockattr_init( &attr );
            pthread_rwlockattr_setpshared( &attr, PTHREAD_PROCESS_SHARED );
            pthread_rwlock_init( &mRWLock, &attr );
            pthread_rwlockattr_destroy( &attr );
        } else {
            pthread_rwlock_init( &mRWLock, NULL );
        }
    }

    inline ~NCRWLock() { pthread_rwlock_destroy( &mRWLock ); }

    inline INT32 readLock() { return -pthread_rwlock_rdlock( &mRWLock ); }

    inline INT32 tryReadLock() { return -pthread_rwlock_tryrdlock( &mRWLock ); }

    inline INT32 writeLock() { return -pthread_rwlock_wrlock( &mRWLock ); }

    inline INT32 tryWriteLock() { return -pthread_rwlock_trywrlock( &mRWLock ); }

    inline void unlock() { pthread_rwlock_unlock( &mRWLock ); }

    class AutoRLock {
       public:
        inline explicit AutoRLock( NCRWLock &rwlock ) : mLock( rwlock ) { mLock.readLock(); }

        inline ~AutoRLock() { mLock.unlock(); }

       private:
        NCRWLock &mLock;
    };

    class AutoWLock {
       public:
        inline explicit AutoWLock( NCRWLock &rwlock ) : mLock( rwlock ) { mLock.writeLock(); }

        inline ~AutoWLock() { mLock.unlock(); }

       private:
        NCRWLock &mLock;
    };

   private:
    pthread_rwlock_t mRWLock;

    NCRWLock( const NCRWLock & );
    NCRWLock &operator=( const NCRWLock & );
};

/**
 * Auto read-lock helper class
 *
 * Create a NCAutoReadLock object to acquire a read-lock until it's destroyed.
 */
class NCAutoReadLock {
   public:
    explicit NCAutoReadLock( NCRWLock &lock ) : m_lock( lock ) { m_lock.readLock(); }

    ~NCAutoReadLock() { m_lock.unlock(); }

   private:
    NCRWLock &m_lock;

    NCAutoReadLock( const NCAutoReadLock & );
    NCAutoReadLock &operator=( const NCAutoReadLock & );
};

/**
 * Auto write-lock helper class
 *
 * Create a NCAutoWriteLock object to acquire a write-lock until it's destroyed.
 */
class NCAutoWriteLock {
   public:
    explicit NCAutoWriteLock( NCRWLock &lock ) : m_lock( lock ) { m_lock.writeLock(); }

    ~NCAutoWriteLock() { m_lock.unlock(); }

   private:
    NCRWLock &m_lock;

    NCAutoWriteLock( const NCAutoWriteLock & );
    NCAutoWriteLock &operator=( const NCAutoWriteLock & );
};
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCRWLOCK_H_
/* EOF */
