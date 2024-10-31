
#pragma once

#include "diag/ipc/adapter/ipc_method_server_adapter.h"

namespace hozon {
namespace netaos {
namespace diag {
class IPCServer {
public:
    IPCServer()
        : adapter_(std::make_unique<IPCMethodServerAdapter>()) {}
    
    virtual ~IPCServer(){};
    

    /** 初始化接口
        @param[in]   domain 域名字
        @param[in]   service_name 服务名字
        @return     int32_t 0为正常，-1为异常
        @note       Service初始化接口
    */
    int32_t Start(const std::string& service_name) {
        adapter_->RegisterProcess([this](const std::vector<uint8_t>& req, std::vector<uint8_t>& resp) {
            return InternalProcess(req, resp);
        });
        return adapter_->Start(service_name);
    }

    /**  析构化接口
    @return     int32_t 0为正常，-1为异常
    @note       Service析构接口
    */
    int32_t Stop() {
        return adapter_->Stop();
    }

    /**  处理函数
    @param[in]   req 请求参数
    @param[out]  resp 回复参数
    @return     int32_t 0为正常，-1为异常
    @note       纯虚函数，需要使用者继承
    */
    virtual int32_t Process(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp) = 0;

private:
    IPCServer(const IPCServer&);
    IPCServer& operator=(const IPCServer&);
    int32_t InternalProcess(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp) {
        return Process(req, resp);
    }

private:
    std::unique_ptr<IPCMethodServerAdapter> adapter_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon