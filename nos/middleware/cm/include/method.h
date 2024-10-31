#pragma once

#include "cm/include/method_adapter.h"
#include "idl/generated/servicebase.h"

namespace hozon {
namespace netaos {
namespace cm {

template <typename ReqType, typename RespType>
class Client {
public:
    explicit Client(
            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
            std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type) :
            _adapter(new MethodClientAdapter(req_topic_type, resp_topic_type)) {

    }

    /**  初始化接口
    @param[in]  service_name 服务名字
    @return     int32_t 0为正常，-1为异常
    @note       Client初始化接口
    */
    int32_t Init(const uint32_t domain, const std::string& service_name) {
        _adapter->RegisterGenResp(std::bind(&Client::GenResp, this));
        _adapter->RegisterAssignmentResp(std::bind(&Client::AssignmentResp, this, std::placeholders::_1, std::placeholders::_2));
        return _adapter->Init(domain, service_name);
    }

    /**  析构化接口
    @return     int32_t 0为正常，-1为异常
    @note       Client析构接口
    */
    int32_t Deinit() {
        return _adapter->Deinit();
    }

    /**  查询服务上线接口
    @param[in]  timeout_ms 最大容忍的服务发现时间,必须大于10ms
    @return     0为上线，-1为失败
    @note       用户需要轮训调用查询服务上线接口
    */
    int32_t WaitServiceOnline(int64_t timeout_ms) {
        return _adapter->WaitServiceOnline(timeout_ms);
    }

    /**  同步请求接口
    @param[in]   req 请求参数
    @param[out]  resp 回复参数
    @param[in]   timeout_ms 超时时间,必须指定超时时间
    @return     int32_t 0为正常，-1为异常
    @note       带回复的同步请求接口
    */
    int32_t Request(std::shared_ptr<ReqType> req, std::shared_ptr<RespType> resp, int64_t timeout_ms) {
        return _adapter->Request(req, resp, timeout_ms);
    }

    /**  异步请求接口
    @param[in]  req 请求参数
    @param[in]  timeout_ms 超时时间,若为0则代表不传入超时参数
    @return     RespType 回复类型
    @note       带回复的异步请求接口,如果返回值为空指针，则代表出现异常
    */
    std::future<std::pair<int32_t, std::shared_ptr<RespType>>> AsyncRequest(std::shared_ptr<ReqType> req, int64_t timeout_ms) {
        return std::async(std::launch::async, [this, req, timeout_ms] {
            std::shared_ptr<RespType> resp = std::make_shared<RespType>();
            int32_t ret = Request(req, resp, timeout_ms);
            std::pair<int32_t, std::shared_ptr<RespType>> pack(ret, resp);
            return pack;
        });
    }

    /**  同步请求接口不带返回值
    @param[in]  req 请求参数
    @return     int32_t 0为正常，-1为异常
    @note       无回复的请求接口
    */
    int32_t RequestAndForget(std::shared_ptr<ReqType> req) {
        return _adapter->RequestAndForget(req);
    }

private:
    std::shared_ptr<ServiceBase> GenResp() {
        return std::make_shared<RespType>();
    }

    void AssignmentResp(std::shared_ptr<ServiceBase>& resp_out, std::shared_ptr<ServiceBase>& resp_in) { *std::static_pointer_cast<RespType>(resp_out) = *std::static_pointer_cast<RespType>(resp_in); }

    std::unique_ptr<MethodClientAdapter> _adapter;
};

template <typename ReqType, typename RespType>
class Server {
public:
    Server(
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> req_topic_type, 
        std::shared_ptr<eprosima::fastdds::dds::TopicDataType> resp_topic_type)
        : _adapter(new MethodServerAdapter(req_topic_type, resp_topic_type)) {}

    /** 初始化接口
        @param[in]   domain 域名字
        @param[in]   service_name 服务名字
        @return     int32_t 0为正常，-1为异常
        @note       Service初始化接口
    */
    int32_t Start(const uint32_t& domain, const std::string& service_name) {
        _adapter->RegisterProcess(std::bind(&Server::InternalProcess, this, std::placeholders::_1, std::placeholders::_2));
        _adapter->RegisterGenReq(std::bind(&Server::GenReq, this));
        _adapter->RegisterGenResp(std::bind(&Server::GenResp, this));
        return _adapter->Start(domain, service_name);
    }

    /**  析构化接口
    @return     int32_t 0为正常，-1为异常
    @note       Service析构接口
    */
    int32_t Stop() {
        return _adapter->Stop();
    }

    /**  处理函数
    @param[in]   req 请求参数
    @param[out]  resp 回复参数
    @return     int32_t 0为正常，-1为异常
    @note       纯虚函数，需要使用者继承
    */
    virtual int32_t Process(const std::shared_ptr<ReqType> req, std::shared_ptr<RespType> resp) = 0;

private:
    int32_t InternalProcess(const std::shared_ptr<ServiceBase> req, std::shared_ptr<ServiceBase> resp) {
        return Process(std::static_pointer_cast<ReqType>(req), std::static_pointer_cast<RespType>(resp));
    }

    std::shared_ptr<ServiceBase> GenReq() {
        return std::make_shared<ReqType>();
    }

    std::shared_ptr<ServiceBase> GenResp() {
        return std::make_shared<RespType>();
    }

    std::unique_ptr<MethodServerAdapter> _adapter;
};
}  // namespace cm
}  // namespace netaos
}  // namespace hozon
