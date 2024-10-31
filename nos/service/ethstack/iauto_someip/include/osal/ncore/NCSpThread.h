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
 * @file NCSpThread.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCSPTHREAD_H
#define NCSPTHREAD_H

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <vector>

#include "osal/ncore/NCAtomic.h"
#include "osal/ncore/NCList.h"
#include "osal/ncore/NCRefBase.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCThread.h"
#include "osal/ncore/NCTimer.h"
#include "osal/ncore/NCTimerThreadPool.h"
#include "osal/ncore/NCWaitObj.h"

OSAL_BEGIN_NAMESPACE
template <typename REQ>
class NCSpThread : public NCThread {
   public:
    NCSpThread( const CHAR* thread_name, NCTimerThreadPool* threadPool );
    virtual ~NCSpThread();

    VOID postRequest( REQ* req );

    typedef typename ncsp<REQ>::sp sp;
    typedef typename ncsp<REQ>::wp wp;
    typedef std::vector<sp>        Vector;

    VOID curRequest( sp& out ) {
        mSyncObj.syncStart();
        out = curReq;
        mSyncObj.syncEnd();
    }

    wp currRequest() { return curReq; }

    ///< wait/notify in Requeset::doAction()
    NC_BOOL isReqAction();
    VOID    notifyRequest();
    VOID    resetNotify();
    NC_BOOL waitRequest( INT32 msec = INFINITE32 );

    ///< override for wakeup mReqWaitObj
    virtual NC_BOOL stopThread( INT32 msec = INFINITE32 );
    virtual VOID    startThread( const CHAR* name = "Unknown Thread" );
    virtual NC_BOOL terminate();

    ///< stop thread after the request::doAction
    virtual NC_BOOL stopThread( REQ* req, NC_BOOL clear, INT32 msec = INFINITE32 );

   protected:
    NC_BOOL popRequest();
    VOID    clearRequest();
    NC_BOOL doStopReq();

    VOID toWait() { mReqWaitObj.waitTime(); }

    VOID toNotify() { mReqWaitObj.notify(); }

   protected:
    sp              curReq;   // for current action.
    sp              stopReq;  // for quit action.
    Vector mReqList GUARDED_BY( mSyncObj );
    NCSyncObj       mSyncObj;
    NCString        mThreadName;

   private:
    ///< override for protect NCThread::waitobj
    virtual VOID notify() { NCThread::notify(); }

    virtual NC_BOOL wait( INT32 msec = INFINITE32 ) { return NCThread::waitTime( msec ); }

    virtual VOID run();

    enum {
        REQ_ACTION = 0x00000010,  // REQ.doAction
    };

    /**
     * @brief
     *
     * @class ReqTimer
     */
    class ReqTimer : public NCTimer {
       public:
        ReqTimer( NCTimerThreadPool* threadPool, INT64 msec, NC_BOOL iter, sp req )
            : NCTimer( threadPool, msec, iter ), curReq( req ) {}

       private:
        sp           curReq;
        virtual VOID onTimer() {
            if ( curReq != NULL ) {
                curReq->doTimeout();
            }
        }
    };

   private:
    NCWaitObj          mReqWaitObj;
    volatile INT32     mReqState;
    NC_BOOL            mQuit;
    NCTimerThreadPool* mThreadPool;
};

template <typename REQ>
NCSpThread<REQ>::NCSpThread( const CHAR* thread_name, NCTimerThreadPool* threadPool )
    : mThreadName( thread_name ), mReqState( 0 ), mQuit( NC_FALSE ), mThreadPool( threadPool ) {
}

template <typename REQ>
NCSpThread<REQ>::~NCSpThread() {
    clearRequest();
    stopThread();
}

template <typename REQ>
VOID NCSpThread<REQ>::postRequest( REQ* req ) {
    mSyncObj.syncStart();
    startThread( mThreadName );
    mReqList.push_back( req );
    mSyncObj.syncEnd();
    toNotify();
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::popRequest() {
    mSyncObj.syncStart();
    if ( 0 != mReqList.size() ) {
        curReq = mReqList[ 0 ];
        mReqList.erase( mReqList.begin() );
        mSyncObj.syncEnd();
        return NC_TRUE;
    }
    curReq = nullptr;  // release the current request.
    mSyncObj.syncEnd();
    return NC_FALSE;
}

template <typename REQ>
VOID NCSpThread<REQ>::clearRequest() {
    mSyncObj.syncStart();
    mReqList.clear();
    mSyncObj.syncEnd();
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::isReqAction() {
    return ( mReqState & REQ_ACTION ) ? NC_TRUE : NC_FALSE;
}

template <typename REQ>
VOID NCSpThread<REQ>::notifyRequest() {
    notify();
}

template <typename REQ>
VOID NCSpThread<REQ>::resetNotify() {
    reset();
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::waitRequest( INT32 msec ) {
    if ( !isReqAction() ) {
        return NC_FALSE;
    }

    if ( INFINITE32 != msec ) {  // start timer
        ReqTimer timer( mThreadPool, msec, false, curReq );
        NC_BOOL  res = timer.start();
        if ( !res ) {
        }
        wait();
    } else {
        wait();
    }
    return NC_TRUE;
}

template <typename REQ>
VOID NCSpThread<REQ>::startThread( const CHAR* name ) {
    mQuit = NC_FALSE;
    return NCThread::startThread( name );
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::stopThread( INT32 msec ) {
    mQuit = NC_TRUE;
    toNotify();  // notify mReqWaitObj
    return NCThread::stopThread( msec );
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::stopThread( REQ* req, NC_BOOL clear, INT32 msec ) {
    mQuit   = NC_TRUE;
    stopReq = req;
    if ( clear ) {  // clear request but current.
        clearRequest();
    }
    toNotify();
    return join( msec );
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::terminate() {
    mQuit = NC_TRUE;
    toNotify();  // notify mReqWaitObj
    return NCThread::terminate();
}

template <typename REQ>
NC_BOOL NCSpThread<REQ>::doStopReq() {
    if ( stopReq != NULL ) {
        curReq = stopReq;
        nc_atomic_or( REQ_ACTION, &mReqState );
        resetNotify();  // reset wait obj, before action...
        curReq->doAction( this );
        nc_atomic_and( ~REQ_ACTION, &mReqState );
        curReq  = NULL;
        stopReq = NULL;

        // stop thread.
        toNotify();
        return stopThread();
    }
    return NC_FALSE;
}

template <typename REQ>
VOID NCSpThread<REQ>::run() {
    while ( !checkQuit() && !mQuit ) {
        while ( !doStopReq() && popRequest() ) {
            nc_atomic_or( REQ_ACTION, &mReqState );
            resetNotify();  // reset wait obj, before action...
            curReq->doAction( this );
            nc_atomic_and( ~REQ_ACTION, &mReqState );
        }
        toWait();
        doStopReq();
    }
}
OSAL_END_NAMESPACE
#endif /* NCSPTHREAD_H */
/* EOF */
