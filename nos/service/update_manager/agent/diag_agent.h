/*
 * Copyright (c) Hozon SOC Co., Ltd. 2022-2022. All rights reserved.
 *
 * Description: ota diag agent module
 */
#ifndef DIAG_AGENT_H
#define DIAG_AGENT_H

#include <stdint.h>
#include <functional>
#include <mutex>
#include <memory>

#include "update_manager/common/data_def.h"
#include "uds_raw_data_req_dispatcher.h"
#include "uds_raw_data_resp_receiver.h"
#include "uds_data_method_receiver.h"
#include "update_status_method_receiver.h"
#include "chassis_info_method_sender.h"
#include "idl/generated/diagPubSubTypes.h"
#include "uds_current_session_receiver.h"

namespace hozon {
namespace netaos {
namespace update {

class DiagAgent {
public:

    static DiagAgent* Instance();

    void Init();
    void Deinit();
    void RegistUdsRawDataReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback);
    void RegistReadVersionReceiveCallback(std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> read_version_receive_callback);
    void DeRegistReadVersionReceiveCallback();

    bool SendUdsRawData(const std::unique_ptr<uds_raw_data_req_t>& uds_raw_data);
    void ReceiveUdsData(const std::shared_ptr<uds_data_req_t>& uds_data_req, std::shared_ptr<uds_data_req_t>& uds_data_resp);
    bool SendChassisInfo(std::unique_ptr<chassis_info_t>& output_info);
    uint8_t GetTotalProgress();
    bool ResetTotalProgress();

private:
    uint8_t UdsDataReceiveCallback(const std::shared_ptr<uds_data_req_t>& uds_data_req,
                                   std::shared_ptr<uds_data_req_t>& uds_data_resp);


    int32_t DidsUpdateProgressDisplayResponse(const std::shared_ptr<uds_data_req_t>& uds_data_req,
                                      std::shared_ptr<uds_data_req_t>& uds_data_resp);  // $22 01 07

private:
    DiagAgent();
    ~DiagAgent();
    DiagAgent(const DiagAgent &);
    DiagAgent & operator = (const DiagAgent &);

    static std::mutex m_mtx;
    static DiagAgent* m_pInstance;

    std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> uds_rawdata_receive_callback_;
    std::function<void(const std::unique_ptr<uds_raw_data_resp_t>&)> read_version_receive_callback_;

    std::unique_ptr<UdsRawDataReqDispatcher> uds_raw_data_req_dispatcher_;
    std::unique_ptr<UdsRawDataRespReceiver> uds_raw_data_resp_receiver_;
    std::unique_ptr<UdsDataMethodReceiver> uds_data_method_receiver_;
    std::unique_ptr<UpdateStatusMethodReceiver> update_status_method_receiver_;
    std::unique_ptr<ChassisInfoMethodSender> chassis_info_method_sender_;
    std::unique_ptr<UdCurrentSessionReceiver> uds_cur_session_receiver_;

    uint8_t progress_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // DIAG_AGENT_H
