/* Copyright (c)  2023 Hozon New Energy Auto Co., Ltd.. All rights reserved.
 *
 * @File: dc_client_impl.h
 * @Date: 2023/10/27
 * @Author: cheng
 * @Desc: --
 */

#pragma once
#ifndef NOS_COMMIT_SERVICE_DATA_COLLECTION_CLIENT_INCLUDE_DC_CLIENT_IMPL_H__
#define NOS_COMMIT_SERVICE_DATA_COLLECTION_CLIENT_INCLUDE_DC_CLIENT_IMPL_H__

#include <vector>

//#include "basic/trans_struct.h"
#include "dc_client.h"
#include "cm/include/method.h"
#include "idl/generated/data_collection_info.h"
#include "idl/generated/data_collection_infoPubSubTypes.h"
#include "log/include/default_logger.h"
//#include "utils/include/dc_logger.hpp"

namespace hozon {
namespace netaos {
namespace dc {


using namespace hozon::netaos::dc;

class DcClientImpl{
   public:
    DcClientImpl() ;

    DcResultCode Init(const std::string client_name, const uint32_t max_wait_millis = 1000);

    DcResultCode DeInit() ;

    DcResultCode CollectTrigger(uint32_t trigger_id);

    DcResultCode CollectTriggerDesc(uint32_t trigger_id, uint64_t time);

    DcResultCode CollectTriggerDesc(uint32_t trigger_id, uint64_t time, std::string desc);

    DcResultCode Upload(std::vector<std::string> &path_list, std::string file_type, std::string file_name, uint16_t cache_file_num);

    DcResultCode Upload(std::vector<char> &data, std::string file_type, std::string file_name, uint16_t cache_file_num);

   private:
    Client<triggerInfo, triggerResult> client_;
    Client<triggerUploadInfo, triggerResult> clientUpload_;
    uint32_t maxWaitTimeMs_{1000};
    std::string clientName_{"unknown"};
    std::atomic_bool isInit_{false};
    std::mutex mtx_;
    std::atomic_bool deleteAfterCompress{false};
};

}  // namespace dc
}  // namespace netaos
}  // namespace hozon

#endif  // NOS_COMMIT_SERVICE_DATA_COLLECTION_CLIENT_INCLUDE_DC_CLIENT_IMPL_H__
