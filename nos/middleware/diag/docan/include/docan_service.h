/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class DocanService Header
 */

#ifndef DOCAN_SERVICE_H_
#define DOCAN_SERVICE_H_
#ifndef __cplusplus
#    error ERROR: This file requires C++ compilation (use a .cpp suffix)
#endif

#include <memory>
#include <string>
#include <vector>
#include "docan_def.h"

namespace hozon {
namespace netaos {
namespace diag {

    class DocanListener;
    class DocanServiceImpl;
    /**
     * @brief class of DocanService.
     *
     * Docan handler class.
     */
    class DocanService
    {
    public:
        DocanService();
        static DocanService* GetDocanService();
        virtual ~DocanService();

        virtual int32_t Init();
        virtual int32_t Start();
        virtual int32_t Stop();
        virtual int32_t Deinit();

        /**
         * @brief 注册/解注册docan的诊断请求的回调listener，用于接收docan请求的应答response和indication
         * @param[in]  who      诊断请求数据
         * @param[in]  listener 诊断请求数据
         * @param[out] none
         * @return     int32_t  若 < 0 为异常
         * @warning    无
         * @note       无
         */
        virtual int32_t registerListener(const std::string& who, const std::shared_ptr<DocanListener>& listener);
        virtual int32_t unregisterListener(const std::string& who);

        /**
         * @brief 通过docan发送诊断请求
         * @param[in]  who    诊断请求的发起方和注册回调的保持一致
         * @param[in]  reqSa  诊断请求client端的逻辑地址
         * @param[in]  reqTa  诊断请求的目标ECU的逻辑地址
         * @param[in]  uds    诊断请求的uds数据
         * @param[out] none
         * @return     int32_t 若< 0 为异常, =0 表示没有注册回调服务  >0 为请求返回的 seqId，可用于匹配应答
         * @warning    无
         * @note       无
         */
        int32_t UdsRequest(const std::string& who, uint16_t reqSa, uint16_t reqTa, const std::vector<uint8_t>& uds);

    private:
        DocanService(const DocanService&);
        DocanService& operator=(const DocanService&);

    private:
        static DocanService* s_instance;
        DocanServiceImpl* m_impl;
    };


} // end of diag
} // end of netaos
} // end of hozon
#endif  // DOCAN_SERVICE_H_
/* EOF */
