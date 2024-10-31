
#pragma once

#include "diag/ipc/adapter/ipc_method_client_adapter.h"

namespace hozon {
namespace netaos {
namespace diag {


class IPCClient {
public:
    IPCClient():adapter_(std::make_unique<IPCMethodClientAdapter>())
    {}

    virtual ~IPCClient(){};

    /**  初始化接口
    @param[in]  service_name 服务名字
    @return     int32_t 0为正常，-1为异常
    @note       Client初始化接口
    */
    int32_t Init(const std::string& service_name) {
        return adapter_->Init(service_name);
    }

    /**  析构化接口
    @return     int32_t 0为正常，-1为异常
    @note       Client析构接口
    */
    int32_t Deinit() {
        return adapter_->Deinit();
    }

    /**  同步请求接口
    @param[in]   req 请求参数
    @param[out]  resp 回复参数
    @param[in]   timeout_ms 超时时间,必须指定超时时间
    @return     int32_t 0为正常，-1为异常
    @note       带回复的同步请求接口
    */
    int32_t Request(const std::vector<uint8_t>& req, std::vector<uint8_t>& resp, const int64_t& timeout_ms) {
        return adapter_->Request(req, resp, timeout_ms);
    }

    /**  异步请求接口
    @param[in]  req 请求参数
    @param[in]  timeout_ms 超时时间,若为0则代表不传入超时参数，一直等待
    @return     std::pair<int32_t, std::vector<uint8_t>>，第一个int32_t是返回值，第二个是resp值
    @note       带回复的异步请求接口,如果返回值为空指针，则代表出现异常
    */
    std::future<std::pair<int32_t, std::vector<uint8_t>>> AsyncRequest(const std::vector<uint8_t>& req, const int64_t& timeout_ms) {
        return std::async(std::launch::async, [this, req, timeout_ms] {
            std::vector<uint8_t> resp;
            int32_t ret = Request(req, resp, timeout_ms);
            std::pair<int32_t, std::vector<uint8_t>> pack(ret, resp);
            return pack;
        });
    }

    /**  同步请求接口不带返回值
    @param[in]  req 请求参数
    @return     int32_t 0为正常，-1为异常
    @note       无回复的请求接口
    */
    int32_t RequestAndForget(const std::vector<uint8_t>& req) {
        return adapter_->RequestAndForget(req);
    }

    /**  异步请求接口
    @param[in]  req 请求参数
    @param[in]  timeout_ms 超时时间,若为0则代表不传入超时参数，一直等待
    @return     int32_t，是接口调用返回值
    @note       带回复的异步请求接口,如果返回值为空指针，则代表出现异常
    */
    std::future<int32_t> AsyncRequestAndForget(const std::vector<uint8_t>& req) {
        return std::async(std::launch::async, [this, req] {
            int32_t ret = RequestAndForget(req);
            return ret;
        });
    }

    /**  客户点端检查服务是否连接成功
    @param[in]  req 请求参数
    @return     int32_t 0为正常，-1为异常
    @note       无回复的请求接口
    */
    int32_t IsMatched() {
        return adapter_->IsMatched();
    }
private:
    IPCClient(const IPCClient&);
    IPCClient& operator=(const IPCClient&);

private:
    std::unique_ptr<IPCMethodClientAdapter> adapter_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
