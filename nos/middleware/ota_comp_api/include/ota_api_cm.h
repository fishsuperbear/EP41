
/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ota api definition
 */

#ifndef HOZON_OTA_API_CM_H_
#define HOZON_OTA_API_CM_H_

#include <iostream>
#include "cm/include/method.h"
#include "idl/generated/devm.h"
#include "idl/generated/devmPubSubTypes.h"
#include "idl/generated/devmTypeObject.h"

namespace hozon {
namespace netaos {
namespace otaapi {

enum State {
    NORMAL_IDLE             = 0x01,
    OTA_PRE_UPDATE          = 0x02,
    OTA_UPDATING            = 0x03,
    OTA_UPDATED             = 0x04,
    OTA_ACTIVING            = 0x05,
    OTA_ACTIVED             = 0x06,
    OTA_UPDATE_FAILED       = 0x07,
};


using namespace hozon::netaos::cm;

class OTAApi {
public:
    OTAApi();
    ~OTAApi();

    void ota_api_init();
    void ota_api_deinit();
    std::string ota_get_version();
    int32_t ota_precheck();
    uint8_t ota_progress();
    int32_t ota_start_update(std::string package_path);
    int32_t ota_get_update_status();

private:
    int32_t GetState(std::string state);

    std::shared_ptr<common_reqPubSubType> req_data_type_status;
    std::shared_ptr<update_status_respPubSubType> resp_data_type_status;
    std::shared_ptr<common_req> req_data_status;
    std::shared_ptr<update_status_resp> resq_data_status;
    Client<common_req, update_status_resp> *client_status;

    std::shared_ptr<common_reqPubSubType> req_data_type_version;
    std::shared_ptr<get_version_respPubSubType> resp_data_type_version;
    std::shared_ptr<common_req> req_data_version;
    std::shared_ptr<get_version_resp> resq_data_version;
    Client<common_req, get_version_resp> *client_version;

    std::shared_ptr<common_reqPubSubType> req_data_type_precheck;
    std::shared_ptr<precheck_respPubSubType> resp_data_type_precheck;
    std::shared_ptr<common_req> req_data_precheck;
    std::shared_ptr<precheck_resp> resq_data_precheck;
    Client<common_req, precheck_resp> *client_precheck;

    std::shared_ptr<common_reqPubSubType> req_data_type_progress = std::make_shared<common_reqPubSubType>();
    std::shared_ptr<progress_respPubSubType> resp_data_type_progress = std::make_shared<progress_respPubSubType>();
    std::shared_ptr<common_req> req_data_progress = std::make_shared<common_req>();
    std::shared_ptr<progress_resp> resq_data_progress = std::make_shared<progress_resp>();
    Client<common_req, progress_resp> *client_progress;

    std::shared_ptr<start_update_reqPubSubType> req_data_type_update;
    std::shared_ptr<start_update_respPubSubType> resp_data_type_update;
    std::shared_ptr<start_update_req> req_data_update;
    std::shared_ptr<start_update_resp> resq_data_update;
    Client<start_update_req, start_update_resp> *client_update;

    uint8_t progress_;
};

}
}
}

#endif
