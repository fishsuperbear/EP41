/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 配置服务
 * Created on: Feb 7, 2023
 *
 */
#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_H_
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <map>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <vector>

#include "cfg_vehiclecfg_parse.h"
#include "cfg_vehiclecfg_update.h"
#include "cm/include/method.h"
#include "cm/include/proxy.h"
#include "cm/include/skeleton.h"
#include "idl/generated/cfg.h"
#include "idl/generated/cfgPubSubTypes.h"
#include "include/cfg_data_def.h"
#include "include/cfg_logger.h"
#include "include/cfg_server_data_def.h"
#include "include/cfg_utils.h"
#include "include/phm_client_instance.h"
namespace hozon {
namespace netaos {
namespace cfg {
// using namespace hozon::netaos::cm;

class CfgServer {
 public:
    class CfgInitClientMethodServer : public hozon::netaos::cm::Server<cfg_initclient_req_method, cfg_initclient_res_method> {
     public:
        CfgInitClientMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_initclient_req_method> req, std::shared_ptr<cfg_initclient_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgInitClientMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgDeInitClientMethodServer : public hozon::netaos::cm::Server<cfg_deinitclient_req_method, cfg_deinitclient_res_method> {
     public:
        CfgDeInitClientMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_deinitclient_req_method> req, std::shared_ptr<cfg_deinitclient_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgDeInitClientMethodServer() {}

     private:
        CfgServer* cfg_server;
    };

    class CfgSetParamMethodServer : public hozon::netaos::cm::Server<cfg_setparam_req_method, cfg_setparam_res_method> {
     public:
        CfgSetParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_setparam_req_method> req, std::shared_ptr<cfg_setparam_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgSetParamMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgGetParamMethodServer : public hozon::netaos::cm::Server<cfg_getparam_req_method, cfg_getparam_res_method> {
     public:
        CfgGetParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_getparam_req_method> req, std::shared_ptr<cfg_getparam_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetParamMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgReSetParamMethodServer : public hozon::netaos::cm::Server<cfg_resetparam_req_method, cfg_resetparam_res_method> {
     public:
        CfgReSetParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_resetparam_req_method> req, std::shared_ptr<cfg_resetparam_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgReSetParamMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgDelParamMethodServer : public hozon::netaos::cm::Server<cfg_delparam_req_method, cfg_delparam_res_method> {
     public:
        CfgDelParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_delparam_req_method> req, std::shared_ptr<cfg_delparam_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgDelParamMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgMonitorParamMethodServer : public hozon::netaos::cm::Server<cfg_monitorparam_req_method, cfg_monitorparam_res_method> {
     public:
        CfgMonitorParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_monitorparam_req_method> req, std::shared_ptr<cfg_monitorparam_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgMonitorParamMethodServer() {}

     private:
        CfgServer* cfg_server;
    };

    class CfgUnMonitorParamMethodServer : public hozon::netaos::cm::Server<cfg_unmonitorparam_req_method, cfg_unmonitorparam_res_method> {
     public:
        CfgUnMonitorParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_unmonitorparam_req_method> req, std::shared_ptr<cfg_unmonitorparam_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgUnMonitorParamMethodServer() {}

     private:
        CfgServer* cfg_server;
    };

    class CfgParamUpdateDataresMethodServer : public hozon::netaos::cm::Server<cfg_paramupdatedatares_req_method, cfg_paramupdatedatares_res_method> {
     public:
        CfgParamUpdateDataresMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_paramupdatedatares_req_method> req, std::shared_ptr<cfg_paramupdatedatares_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgParamUpdateDataresMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgGetMonitorClientsMethodServer : public hozon::netaos::cm::Server<cfg_getmonitorclients_req_method, cfg_getmonitorclients_res_method> {
     public:
        CfgGetMonitorClientsMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_getmonitorclients_req_method> req, std::shared_ptr<cfg_getmonitorclients_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetMonitorClientsMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    class CfgGetParamInfoListMethodServer : public hozon::netaos::cm::Server<cfg_getparaminfolist_req_method, cfg_getparaminfolist_res_method> {
     public:
        CfgGetParamInfoListMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServer* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<cfg_getparaminfolist_req_method> req, std::shared_ptr<cfg_getparaminfolist_res_method> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetParamInfoListMethodServer() {}

     private:
        CfgServer* cfg_server;
    };
    CfgServer()
        : cfg_init_client_method_server_(std::make_shared<cfg_initclient_req_methodPubSubType>(), std::make_shared<cfg_initclient_res_methodPubSubType>(), this),
          cfg_deinit_client_method_server_(std::make_shared<cfg_deinitclient_req_methodPubSubType>(), std::make_shared<cfg_deinitclient_res_methodPubSubType>(), this),
          cfg_setparam_method_server_(std::make_shared<cfg_setparam_req_methodPubSubType>(), std::make_shared<cfg_setparam_res_methodPubSubType>(), this),
          cfg_getparam_method_server_(std::make_shared<cfg_getparam_req_methodPubSubType>(), std::make_shared<cfg_getparam_res_methodPubSubType>(), this),
          cfg_resetparam_method_server_(std::make_shared<cfg_resetparam_req_methodPubSubType>(), std::make_shared<cfg_resetparam_res_methodPubSubType>(), this),
          cfg_delparam_method_server_(std::make_shared<cfg_delparam_req_methodPubSubType>(), std::make_shared<cfg_delparam_res_methodPubSubType>(), this),
          cfg_monitorparam_method_server_(std::make_shared<cfg_monitorparam_req_methodPubSubType>(), std::make_shared<cfg_monitorparam_res_methodPubSubType>(), this),
          cfg_unmonitorparam_method_server_(std::make_shared<cfg_unmonitorparam_req_methodPubSubType>(), std::make_shared<cfg_unmonitorparam_res_methodPubSubType>(), this),
          cfg_paramupdatedatares_method_server_(std::make_shared<cfg_paramupdatedatares_req_methodPubSubType>(), std::make_shared<cfg_paramupdatedatares_res_methodPubSubType>(), this),
          cfg_getmonitorclients_method_server_(std::make_shared<cfg_getmonitorclients_req_methodPubSubType>(), std::make_shared<cfg_getmonitorclients_res_methodPubSubType>(), this),
          cfg_getparaminfolist_method_server_(std::make_shared<cfg_getparaminfolist_req_methodPubSubType>(), std::make_shared<cfg_getparaminfolist_res_methodPubSubType>(), this),
          cfg_paramupdatedatares_event_proxy_(std::make_shared<cfg_paramupdatedatares_eventPubSubType>()) {
        g_stopFlag = false;
        cfg_paramupdatedata_event_skeleton_map_.clear();
    }

    ~CfgServer() {
        g_stopFlag = false;
        cfg_paramupdatedata_event_skeleton_map_.clear();
    }
    void Init(std::string dir_path, std::string redundant_path, uint32_t maxcom_vallimit);
    void Run();
    void DeInit();
    int32_t RequestProcess(const std::shared_ptr<cfg_initclient_req_method> req, std::shared_ptr<cfg_initclient_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_deinitclient_req_method> req, std::shared_ptr<cfg_deinitclient_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_setparam_req_method> req, std::shared_ptr<cfg_setparam_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_getparam_req_method> req, std::shared_ptr<cfg_getparam_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_resetparam_req_method> req, std::shared_ptr<cfg_resetparam_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_delparam_req_method> req, std::shared_ptr<cfg_delparam_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_monitorparam_req_method> req, std::shared_ptr<cfg_monitorparam_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_unmonitorparam_req_method> req, std::shared_ptr<cfg_unmonitorparam_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_paramupdatedatares_req_method> req, std::shared_ptr<cfg_paramupdatedatares_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_getmonitorclients_req_method> req, std::shared_ptr<cfg_getmonitorclients_res_method> resp);
    int32_t RequestProcess(const std::shared_ptr<cfg_getparaminfolist_req_method> req, std::shared_ptr<cfg_getparaminfolist_res_method> resp);

 protected:
    CfgServerData cfgServerData_;
    CfgVehicleUpdate CfgVehicleImpl_;
    std::shared_timed_mutex cfgServerDataMutex;
    bool g_stopFlag;
    int cmres = -1;

 private:
    CfgInitClientMethodServer cfg_init_client_method_server_;
    CfgDeInitClientMethodServer cfg_deinit_client_method_server_;
    CfgSetParamMethodServer cfg_setparam_method_server_;
    CfgGetParamMethodServer cfg_getparam_method_server_;
    CfgReSetParamMethodServer cfg_resetparam_method_server_;
    CfgDelParamMethodServer cfg_delparam_method_server_;
    CfgMonitorParamMethodServer cfg_monitorparam_method_server_;
    CfgUnMonitorParamMethodServer cfg_unmonitorparam_method_server_;
    CfgParamUpdateDataresMethodServer cfg_paramupdatedatares_method_server_;
    CfgGetMonitorClientsMethodServer cfg_getmonitorclients_method_server_;
    CfgGetParamInfoListMethodServer cfg_getparaminfolist_method_server_;
    std::map<std::string, std::shared_ptr<hozon::netaos::cm::Skeleton>> cfg_paramupdatedata_event_skeleton_map_;
    hozon::netaos::cm::Proxy cfg_paramupdatedatares_event_proxy_;
    void SubServerData(const CfgServerData& strdata, CfgServerData& dstdata);
    void ParamUpdateDataResEventCallback();
    void Persist(string key);
};
};  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_H_
