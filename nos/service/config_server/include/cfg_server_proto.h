/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: per
 * Description: 配置服务
 * Created on: Feb 7, 2023
 *
 */
#ifndef SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_PROTO_H_
#define SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_PROTO_H_
#include <sys/types.h>
#include <unistd.h>

#include <functional>
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
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "idl/generated/cfg.h"
#include "idl/generated/cfgPubSubTypes.h"
#include "idl/generated/cm_protobuf.h"
#include "idl/generated/cm_protobufPubSubTypes.h"
#include "idl/generated/proto_method.h"
#include "idl/generated/proto_methodPubSubTypes.h"
#include "include/cfg_data_def.h"
#include "include/cfg_logger.h"
#include "include/cfg_server_data_def.h"
#include "include/cfg_utils.h"
#include "include/phm_client_instance.h"

#include "cfg.pb.h"
namespace hozon {
namespace netaos {
namespace cfg {
// using namespace hozon::netaos::cm;

class CfgServerProto {
 public:
    CfgServerProto();
    ~CfgServerProto();
    class CfgInitClientMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgInitClientMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) {
            cfg_server->RequestProcess(req, resp);
            return 0;
        }
        ~CfgInitClientMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgDeInitClientMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgDeInitClientMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgDeInitClientMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };

    class CfgSetParamMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgSetParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgSetParamMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgGetParamMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgGetParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetParamMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgReSetParamMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgReSetParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgReSetParamMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgDelParamMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgDelParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgDelParamMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgMonitorParamMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgMonitorParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgMonitorParamMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };

    class CfgUnMonitorParamMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgUnMonitorParamMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgUnMonitorParamMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };

    class CfgParamUpdateDataresMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgParamUpdateDataresMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgParamUpdateDataresMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgGetMonitorClientsMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgGetMonitorClientsMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetMonitorClientsMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgGetParamInfoListMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgGetParamInfoListMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetParamInfoListMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class CfgGetClientInfoListMethodServer : public hozon::netaos::cm::Server<ProtoMethodBase, ProtoMethodBase> {
     public:
        CfgGetClientInfoListMethodServer(std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_data, std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_data, CfgServerProto* server)
            : Server(req_data, resp_data), cfg_server(server) {}
        int32_t Process(const std::shared_ptr<ProtoMethodBase> req, std::shared_ptr<ProtoMethodBase> resp) { return cfg_server->RequestProcess(req, resp); }
        ~CfgGetClientInfoListMethodServer() {}

     private:
        CfgServerProto* cfg_server;
    };
    class FunctionMapper {
     public:
        template <typename... Args>
        void addFunction(const std::string& str, const std::function<void(Args...)>& function) {
            functionMap[str] = function;
        }
        template <typename... Args>
        void callFunction(const std::string& str, Args... args) {
            auto function = functionMap[str];
            if (function) {
                function(args...);
            } else {
                CONFIG_LOG_WARN << "Unknown function! " << str.c_str();
            }
        }

     private:
        std::map<std::string, std::function<void(const std::string, std::shared_ptr<ProtoMethodBase>)>> functionMap;
    };

    void Init(std::string dir_path, std::string redundant_path, uint32_t maxcom_vallimit);
    void Run();
    void DeInit();
    int32_t RequestProcess(const std::shared_ptr<ProtoMethodBase> req_idl, std::shared_ptr<ProtoMethodBase> resp_idl);

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
    CfgGetClientInfoListMethodServer cfg_getclientinfolist_method_server_;
    std::map<std::string, std::shared_ptr<hozon::netaos::cm::Skeleton>> cfg_paramupdatedata_event_skeleton_map_;
    hozon::netaos::cm::Proxy cfg_paramupdatedatares_event_proxy_;
    void SubServerData(const CfgServerData& strdata, CfgServerData& dstdata);
    void ParamUpdateDataResEventCallback();
    void Persist(string key);
    FunctionMapper functionMapper;
    void InitclientReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void DeinitclientReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void SetparamReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void GetparamReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void MonitorparamReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void UnmonitorparamReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void ResetparamReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void GetmonitorclientsReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void GetparaminfolistReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void GetclientinfolistReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void DelparamReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
    void ParamupdatedataresReqMethod(const std::string deserialized_string, std::shared_ptr<ProtoMethodBase> resp_idl);
};
};  // namespace cfg
}  // namespace netaos
}  // namespace hozon
#endif  // SERVICE_CONFIG_SERVER_INCLUDE_CFG_SERVER_PROTO_H_
