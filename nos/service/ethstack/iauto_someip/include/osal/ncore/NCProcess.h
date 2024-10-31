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
 * @file NCProcess.h
 * @brief
 * @date 2020-05-08
 *
 */

#ifndef INCLUDE_NCORE_NCPROCESS_H_
#define INCLUDE_NCORE_NCPROCESS_H_
#ifndef __cplusplus
#error ERROR: This file requires C++ compilation(use a .cpp suffix)
#endif

#include "osal/ncore/NCGlobalAPI.h"
#include "osal/ncore/NCString.h"
#include "osal/ncore/NCThreadSystem.h"
#include "osal/ncore/NCTypesDefine.h"

/**
 * @defgroup Process How to use NCProcess and rpc
 * @{
 *  NCProcess is base class of NCApplicationProcess, NCDaemonProcess,
 *NCServerProcess.
 *  Normally you should inherit NCXXXProcess to implement your own process,
 *these process
 *  will start their specialize rpc and event loop
 *  @par Sample code of NCServerProcess usage
 *  @code
 *
 *  #define PROCESS_NAME "xxxServer"
 *
 *  class xxxServer: public NCServerProcess
 *  {
 *  public:
 *      xxxServer(INT32 argc, CHAR** argv)
 *      : NCServerProcess(PROCESS_NAME, argc, argv)
 *      {}
 *  private:
 *      // do initialize, start, stop, deinitialize of
 *      // you own precess
 *      virtual VOID OnInitialize();
 *      virtual VOID OnStart();
 *      virtual VOID OnStop();
 *      virtual VOID OnDeinitialize();
 *      }
 *  }
 *
 *  INT32 main(INT32 argc, char** argv)
 *  {
 *      xxxServer xxx(argc, argv);
 *
 *      xxx.initialize(new xxxServiceFac());
 *
 *      xxx.start();
 *
 *      xxx.enterloop();
 *
 *      xxx.stop();
 *
 *      xxx.deinitialize();
 *
 *      return 0;
 *   }
 *  @endcode
 *
 *  @par Sample code of rpc howto
 *  @code
 *  //Server side
 *  //implement the onTransact() function for received packages
 *
 *  //service name, should be unique in the whole OS
 *  #define XXX_SERVICE_NAME "nutshell.xxxService"
 *  //rpc code, should be unique in the service
 *  #define XXX_FUNCID_SETN 100
 *
 *  class xxxService: public NCService
 *  {
 *  public:
 *      xxxService() : NCService(XXX_SERVICE_NAME) {}
 *      ~xxxService() {}
 *
 *      // do initialize, start, stop, deinitialize of
 *      // you own service
 *      virtual VOID initialize();
 *      virtual VOID start();
 *      virtual VOID stop();
 *      virtual VOID deinitialize();
 *
 *      NC_BOOL onTransact(uint32_t code, const NCParcel& data, NCParcel* reply,
 *uint32_t flags)
 *      {
 *           // this is the rpc call response function
 *           switch code{
 *           case XXX_FUNCID_SETN:
 *               // unpack data, do some thing with data
 *               // write return values to reply
 *               INT32 i = data.readInt32();
 *               i = i*356+25;
 *               if (reply) reply->writeInt32(i);
 *               return NC_TRUE;
 *           default:
 *               return NC_FALSE;
 *           }
 *      }
 *  };
 *
 *  //in the Service factory, we control all Services
 *  class xxxServiceFac: public NCServiceFactory
 *  {
 *  public:
 *      // to create all services
 *      virtual VOID createService();
 *      {
 *          m_xxxservice = new xxxService();
 *          m_xxxservice->registerService();
 *          m_yyyservice = new yyyService();
 *          m_yyyservice->registerService();
 *      }
 *      // to unregitser all services (not receive rpc calling)
 *      virtual VOID unregisterService();
 *      {
 *          if (m_xxxservice!=NULL) m_xxxservice->unregisterService();
 *          if (m_yyyservice!=NULL) m_yyyservice->unregisterService();
 *      }
 *      // to initialize, start, stop, deinitialize all services
 *      virtual VOID initialize();
 *      {
 *          if (m_xxxservice!=NULL) m_xxxservice->initialize();
 *          if (m_yyyservice!=NULL) m_yyyservice->initialize();
 *      }
 *      virtual VOID start();
 *      {
 *          if (m_xxxservice!=NULL) m_xxxservice->start();
 *          if (m_yyyservice!=NULL) m_yyyservice->start();
 *      }
 *      virtual VOID stop();
 *      {
 *          if (m_xxxservice!=NULL) m_xxxservice->stop();
 *          if (m_yyyservice!=NULL) m_yyyservice->stop();
 *      }
 *      virtual VOID deinitialize();
 *      {
 *          if (m_xxxservice!=NULL) m_xxxservice->deinitialize();
 *          if (m_yyyservice!=NULL) m_yyyservice->deinitialize();
 *      }
 *  private:
 *      ncsp<xxxService>::sp m_xxxservice;
 *      ncsp<yyyService>::sp m_yyyservice;
 *  }
 *
 *  // do set service factory in the main function
 *  INT32 main(INT32 argc, char** argv)
 *  {
 *      xxxServer xxx(argc, argv);
 *      .......
 *      xxx.initialize(new xxxServiceFac());
 *      .......
 *
 *      return 0;
 *   }
 *
 *  //in the client side
 *  class Xxx: public NCServiceProxyBase
 *  {
 *  public:
 *      Xxx() : NCServiceProxyBase(XXX_SERVICE_NAME) {}
 *      ~Xxx() {}
 *
 *      // i/f
 *      INT32 setN(INT32 n) {
 *              //pack the parameters
 *              NCParcel data, reply;
 *              data.writeInt32(n);
 *
 *              //do the rpc call
 *              INT32 ret = transactService(XXX_FUNCID_SETN, data, reply);
 *              //if ret == -1, means transaction error occur
 *
 *              //get the return value, then return it
 *              INT32 i = reply.readInt32();
 *              return i;
 *      }
 *  };
 *
 *  // server response calling from client side
 *  NC_BOOL
 *  xxxService::onTransact(uint32_t code, const NCParcel& data, NCParcel* reply,
 *uint32_t flags)
 *  {
 *      // this is the rpc call response function
 *      switch code{
 *      case XXX_FUNCID_SETN:
 *          // unpack data, do some thing with data
 *          // write return values to reply
 *          INT32 i = data.readInt32();
 *          i = i*356+25;
 *          if (reply) reply->writeInt32(i);
 *          return NC_TRUE;
 *      .......
 *   }
 *
 *  // client call the rpc
 *  Xxx x;
 *  x.setN(10);
 *
 *  @endcode
 *  @}
 *
 **/

OSAL_BEGIN_NAMESPACE
class NCMainLooper;

/**
 *  @class NCProcess
 *
 *  @brief The base class of the process
 *
 *  Which is inherited by : \n
 *  NCApplicationProcess : for normal apl progress \n
 *  NCServerProcess      : for server progress \n
 *  NCDaemonProcess      : for deamon progress \n
 */
class __attribute__( ( visibility( "default" ) ) ) NCProcess {
   public:
    struct NCProcessParam {
        INT32 io_threads;  // io_threads of nc_event_sys in current process
    };

    enum PROCESS_PRIO {
        NC_PROCESS_APPLICATION_PRIO = -4,
        NC_PROCESS_SERVER_PRIO      = 0,
        NC_PROCESS_DAEMON_PRIO      = -6,
    };

    /**
     * @brief Construction of NCProcess
     *
     * @param name
     * - Process name
     * @param argc
     * - arguments number
     * @param argv
     * - arguments vector
     */
    NCProcess( const CHAR *const name, INT32 argc, const CHAR *const *argv );

    /**
     * @brief Construction of NCProcess
     *
     * @param argc
     * - arguments number
     * @param argv
     * - arguments vector
     */
    NCProcess( INT32 argc, const CHAR *const *argv );

    /**
     * @brief Destruction of NCProcess
     */
    virtual ~NCProcess();

    /**
     * @brief initialize event, rpc etc. service
     */
    virtual VOID initialize();

    /**
     * @brief start event, and other user defined actions
     * @param factory
     * - services which is needed to start
     */
    virtual VOID start();

    /**
     * @brief stop event, and other user defined actions
     */
    virtual VOID stop();

    /**
     * @brief destory event system and other user defined actions
     */
    virtual VOID deinitialize();

    /**
     * @brief start rpc listening thread loop
     * @param isMain
     * - NC_TRUE : rpc running in the main thread
     * - NC_FALSE: rpc not running in the main thread
     */
    virtual INT32 enterloop();

    /**
     * @brief quit the executable softly
     */
    static VOID quit( INT32 exitcode = 0 );

    /**
     * @brief set process priority
     */
    static VOID setPriority( INT32 prio );

   protected:
    /**
     * @brief callback , which is called after Initialize
     */
    virtual VOID OnInitialize() = 0;

    /**
     * @brief callback , which is called after Start
     */
    virtual VOID OnStart() = 0;

    /**
     * @brief callback , which is called after stop
     */
    virtual VOID OnStop() = 0;

    /**
     * @brief callback , which is called after Deinitialize
     */
    virtual VOID OnDeinitialize() = 0;

    /**
     * @brief callback , which is called to customize process parameter
     */
    virtual VOID OnConfigure( NCProcessParam &param );

    /**
     * @brief callback , which is called to customize process parameter
     */
    virtual NC_BOOL addThreadTable( const NC_THREAD_TABLE *const table );

    /**
     * @brief set main thread loop
     */
    VOID setMainThreadLoop() const;

   protected:
    NCString       tag;             // < tag name
    NCString       mName;           // < process name
    UINT32         mStartTime;      // < process start time
    NC_BOOL        mRpcLoop;        // < entered main loop
    NCMainLooper * m_mainLooper;    // < inner implementation
    NCThreadSystem m_threadSystem;  // < thread table system

   private:
    INT32 startMainLoop();
    NCProcess( const NCProcess & );
    NCProcess &operator=( const NCProcess & );
};
/** @}*/
OSAL_END_NAMESPACE
#endif  // INCLUDE_NCORE_NCPROCESS_H_
/* EOF */
