
#ifndef DOSOMEIP_TRANSPORT_H
#define DOSOMEIP_TRANSPORT_H

#include <stdint.h>
#include <functional>
#include <mutex>

#include "diag/dosomeip/manager/dosomeip_manager.h"

namespace hozon {
namespace netaos {
namespace diag {
class DoSomeIPTransport {
public:
    static DoSomeIPTransport* getInstance();

    virtual ~DoSomeIPTransport(){};

    /**  DoSomeIPTransport 初始化SomeIP协议栈, 负责创建基于Someip的连接通讯，注册回调函数
     @param[in]  uds_request_callback    收到someip客户端UDS消息的回调函数, 注意这个回调函数不能组阻塞，需要立刻返回
     @param[in]  someip_register_callback   客户端/服务端someip建立连接的回调函数，初始化完成即触发回调
     @param[out] none
     @return     DOSOMEIP_RESULT ：参考dosomeip_def.h定义
     @warning    无
     @note       无
    */
    DOSOMEIP_RESULT DosomeipInit(std::function<void(const DoSomeIPReqUdsMessage&)> uds_request_callback, std::function<void(const DOSOMEIP_REGISTER_STATUS&)> someip_register_callback);

    /**  DoSomeIPTransport 反初始化SomeIP协议栈, 负责断开Someip的连接
     @param[in]  none
     @param[out] none
     @return     无
     @warning    无
     @note       无
    */
    void DosomeipDeinit();

    /**  DoSomeIPTransport TPL层处理完成UDS请求，获得UDS诊断结果后主动调用此接口
     @param[in]  udsMsg UDS消息处理完成的响应信息 
     @param[out] none
     @return     DOSOMEIP_RESULT ：参考dosomeip_def.h定义
     @warning    无
     @note       
    */
    DOSOMEIP_RESULT ReplyUdsOnSomeIp(const DoSomeIPRespUdsMessage& udsMsg);

private:
    DoSomeIPTransport();
    DoSomeIPTransport(const DoSomeIPTransport&);
    DoSomeIPTransport& operator=(const DoSomeIPTransport&);

private:
    static DoSomeIPTransport* instance_;
    static std::mutex mtx_;
    std::unique_ptr<DoSomeIPManager> someip_mgr_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // DOSOMEIP_TRANSPORT_H