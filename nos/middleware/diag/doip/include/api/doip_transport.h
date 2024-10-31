/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description: doip transport
 */

#ifndef MIDDLEWARE_DIAG_DOIP_INCLUDE_API_DOIP_TRANSPORT_H_
#define MIDDLEWARE_DIAG_DOIP_INCLUDE_API_DOIP_TRANSPORT_H_

#include <stdint.h>

#include <functional>
#include <memory>

#include "diag/doip/include/handler/doip_socket_handler.h"
#include "diag/doip/include/api/doip_def.h"

namespace hozon {
namespace netaos {
namespace diag {


class DoIPTransport {
 public:
    DoIPTransport();
    ~DoIPTransport();

    /**  DoIPTransport 初始化DoIP协议栈
     @param[in]  indication_callback 诊断数据接收的回调
     @param[in]  confirm_callback    诊断请求/回应发送结果的回调
     @param[in]  route_callback      诊断路由的回调,DoIP转DoCAN的场景
     @param[out] none
     @return     doip_result_t 参考doip_def.h定义
     @warning    初始化成功后，才能调用诊断数据发送接口
     @note       无
    */
    doip_result_t DoipInit(std::function<void(doip_indication_t*)> indication_callback,
                           std::function<void(doip_confirm_t*)>    confirm_callback,
                           std::function<void(doip_route_t*)>      route_callback,
                           std::string doip_config = "");

    /**  DoIPTransport 反初始化DoIP协议栈
     @param[in]  none
     @param[out] none
     @return     
     @warning    无
     @note       无
    */
    void DoipDeinit();

    /**  DoIPTransport 注册客户端断连通知
     @param[in]  release_callback 断连通知回调，参数1：doip网卡状态 参数2：断连的客户端逻辑地址
     @param[out] none
     @return     
     @warning    无
     @note       用于清除客户端资源
    */
    void DoipRegistReleaseCallback(std::function<void(doip_netlink_status_t, uint16_t)> release_callback);

    /**  DoIPTransport doip_server的诊断请求发送，通常用于被诊断节点的UDS回应
     @param[in]  request 诊断请求数据
     @param[out] none
     @return     doip_result_t 参考doip_def.h定义
     @warning    无
     @note       无
    */
    doip_result_t DoipRequestByNode(const doip_request_t* request);

    /**  DoIPTransport doip_client的诊断请求发送，通常用于诊断测试设备的UDS请求
     @param[in]  request 诊断请求数据
     @param[out] none
     @return     doip_result_t 参考doip_def.h定义
     @warning    无
     @note       无
    */
    doip_result_t DoipRequestByEquip(const doip_request_t* request);

    /**  DoIPTransport doip_client的断连请求发送，通常用于主动断开和doip的连接
     @param[in]  request 断连请求数据
     @param[out] none
     @return     doip_result_t 参考doip_def.h定义
     @warning    无
     @note       无
    */
    doip_result_t DoipReleaseByEquip(const doip_request_t* request);

 private:
    std::shared_ptr<DoIPSocketHandler> socket_handler_;
};

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
#endif  // MIDDLEWARE_DIAG_DOIP_INCLUDE_API_DOIP_TRANSPORT_H_
