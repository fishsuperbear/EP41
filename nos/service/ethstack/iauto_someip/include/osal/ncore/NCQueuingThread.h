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
 * @file NCQueuingThread.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef NCQUEUINGTHREAD_H
#define NCQUEUINGTHREAD_H

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCList.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCSyncObj.h"
#include "osal/ncore/NCThread.h"
#include "osal/ncore/NCTypesDefine.h"

#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

OSAL_BEGIN_NAMESPACE
/**
 * @brief
 *
 * @class NCQueuingThread
 *
 * @code
 *
 * class MyReq
 * {
 * public:
 *     VOID doAction(NCQueuingThread<MyReq>* pThread)
 *     {
 *     //do something
 *     }
 * };
 *
 * class CSample {
 *     NCQueuingThread<MyReq> mThread;
 * public:
 *     CSample() : mThread(NCTEXT("SAMPLE_TRHEAD_NAME")) {}
 *     VOID addReq()
 *     {
 *         MyReq* req = new MyReq();
 *         mThread.postRequest(req);
 *     }
 * };
 *
 * class CSample
 * {
 *     class MyThread : public NCQueuingThread<MyReq>
 *     {
 *     public:
 *         MyThread() : NCQueuingThread<MyReq>(NCTEXT("SAMPLE_TRHEAD_NAME")) {}
 *     };
 *
 *     MyThread mThread;
 * public:
 *     VOID addReq()
 *     {
 *         MyReq* req = new MyReq();
 *         mThread.postRequest(req);
 *     }
 * };
 * @endcode
 */
template <typename REQ>
class NCQueuingThread : public NCThread {
   public:
    /**
     * @brief Construct a new NCQueuingThread
     *
     * @param thread_name name of thread
     */
    NCQueuingThread( const CHAR *thread_name ) : mThreadName( thread_name ) {}

    /**
     * @brief Destroy the NCQueuingThread
     */
    virtual ~NCQueuingThread() {
        mSyncObj.syncStart();
        mReqList.clearData();
        mSyncObj.syncEnd();
        stopThread();
    }

    /**
     * @brief post the request to the end of current List
     *
     * @param rep request
     */
    virtual VOID postRequest( REQ *rep );

    /**
     * @brief push the request to the begin of current List
     *
     * @param rep request
     */
    virtual VOID pushRequest( REQ *rep );

   protected:
    virtual VOID run();
    virtual REQ *popRequest();

    NCList<REQ> mReqList GUARDED_BY( mSyncObj );
    NCSyncObj            mSyncObj;
    NCString             mThreadName;
};

template <typename REQ>
VOID NCQueuingThread<REQ>::postRequest( REQ *rep ) {
    mSyncObj.syncStart();
    startThread( mThreadName );
    mReqList.append( rep );
    mSyncObj.syncEnd();
    notify();
}

template <typename REQ>
VOID NCQueuingThread<REQ>::pushRequest( REQ *rep ) {
    mSyncObj.syncStart();
    startThread( mThreadName );
    mReqList.push( rep );
    mSyncObj.syncEnd();
    notify();
}

template <typename REQ>
REQ *NCQueuingThread<REQ>::popRequest() {
    mSyncObj.syncStart();
    REQ *req = mReqList.pop();
    mSyncObj.syncEnd();
    return req;
}

template <typename REQ>
VOID NCQueuingThread<REQ>::run() {
    REQ *req = NULL;
    while ( !checkQuit() ) {
        req = popRequest();
        while ( NULL != req ) {
            req->doAction( this );
            delete req;
            req = popRequest();
        }
        waitTime();
    }
}
OSAL_END_NAMESPACE

#endif /* NCQUEUINGTHREAD_H */
/* EOF */
