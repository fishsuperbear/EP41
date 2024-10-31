/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanListener Header
 */

#ifndef DOCAN_LISTENER_H_
#define DOCAN_LISTENER_H_

#include <vector>
#include "docan_def.h"

namespace hozon {
namespace netaos {
namespace diag {

class DocanListener {
public:
    DocanListener() {}
    virtual ~DocanListener() {}

public:

    /**
     * @brief 回调接口，回调通知UDS请求的response/indication
     * @param[in]  reqSa  诊断请求的目标ECU的逻辑地址
     * @param[in]  reqTa  诊断请求client端的逻辑地址
     * @param[in]  reqid  诊断请求调用成功返回的reqId
     * @param[in]  result 诊断请求的结果，定义参照docan_def.h
     * @param[in]  uds 诊断请求的uds数据
     * @param[out] none
     * @return     void
     * @warning    无
     * @note       无
     */
    virtual void OnUdsResponse(uint16_t reqSa, uint16_t reqTa, uint32_t reqId, docan_result_t result, const std::vector<uint8_t>& uds) {}
    virtual void OnUdsIndication(uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds) {}

    /**
     * @brief 回调接口，回调通知注册的listener服务bind/unbind，
     * @param[in]  name   注册服务请求的调用方
     * @param[out] none
     * @return     void
     * @warning    无
     * @note       无
     */
    virtual void onServiceBind(const std::string& name) {}
    virtual void onServiceUnbind(const std::string& name) {}

private:
    DocanListener(const DocanListener &);
    DocanListener & operator = (const DocanListener &);

};

} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_LISTENER_H_