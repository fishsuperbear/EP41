/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ota api definition
 */
#include "ota_api_zmq.h"
#include "ota_api_logger.h"
#include "zmq_ipc/proto/devm_tool.pb.h"

namespace hozon {
namespace netaos {
namespace otaapi {


OTAApiZmq::OTAApiZmq() : progress_(0)
{
    client_status_ = std::make_shared<ZmqIpcClient>();
    client_precheck_ = std::make_shared<ZmqIpcClient>();
    client_progress_ = std::make_shared<ZmqIpcClient>();
    client_update_ = std::make_shared<ZmqIpcClient>();
    client_version_ = std::make_shared<ZmqIpcClient>();
}

OTAApiZmq::~OTAApiZmq()
{
}

void
OTAApiZmq::ota_api_init()
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_api_init";
    client_status_->Init("tcp://localhost:11130");
    client_precheck_->Init("tcp://localhost:11131");
    client_progress_->Init("tcp://localhost:11132");
    client_update_->Init("tcp://localhost:11133");
    client_version_->Init("tcp://localhost:11134");
}

void
OTAApiZmq::ota_api_deinit()
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_api_deinit";
    client_status_->Deinit();
    client_precheck_->Deinit();
    client_progress_->Deinit();
    client_update_->Deinit();
    client_version_->Deinit();
}

std::string
OTAApiZmq::ota_get_version()
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_get_api_version";

    std::string ret_version{};
    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_version_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        OTA_API_LOG_ERROR << "version request ret err.";
        return "";
    }

    UpgradeVersionResp resp{};
    resp.ParseFromString(reply);
    if (resp.error_code() > 0) {
        OTA_API_LOG_WARN << "get api version resp error: " << resp.error_code() << ", " << resp.error_msg();
        ret_version = "";
    }
    else {
        OTA_API_LOG_INFO << "major_version: " << resp.major_version();
        OTA_API_LOG_INFO << "soc_version: " << resp.soc_version();
        OTA_API_LOG_INFO << "mcu_version: " << resp.mcu_version();
        OTA_API_LOG_INFO << "sensor_version: ";
        for (const auto& pair : resp.sensor_version()) {
            OTA_API_LOG_INFO << "    " << pair.first << ": " << pair.second;
        }
        ret_version = resp.major_version();
    }

    return ret_version;
}

int32_t
OTAApiZmq::ota_precheck()
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_precheck";

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_precheck_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        OTA_API_LOG_ERROR << "precheck request ret err.";
        return -1;
    }

    UpgradePrecheckResp precheck{};
    precheck.ParseFromString(reply);
    if (precheck.error_code() > 0) {
        OTA_API_LOG_WARN << "precheck resp error: " << precheck.error_code() << ", " << precheck.error_msg();
        return -2;
    }
    else {
        if (precheck.space() == true 
            && precheck.speed() == true 
            && precheck.gear() == true) {
            OTA_API_LOG_INFO << "precheck resp true.";
        }
        else {
            OTA_API_LOG_WARN << "precheck resp false," \
                << " space:" << precheck.space() \
                << " speed:" << precheck.speed() \
                << " gear:" << precheck.gear();
            return -3;
        }
    }

    return 0;
}
uint8_t
OTAApiZmq::ota_progress()
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_progress";

    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_progress_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        OTA_API_LOG_ERROR << "progress request ret err.";
        return progress_;
    }

    UpgradeProgressResp resp{};
    resp.ParseFromString(reply);
    if (resp.error_code() > 0) {
        OTA_API_LOG_WARN << "progress resp error: " << resp.error_code() << ", " << resp.error_msg();
        return progress_;
    }
    else {
        OTA_API_LOG_INFO << "progress resp upgrade progress: " << resp.progress() << "%";
    }

    progress_ = resp.progress();
    return progress_;
}
int32_t
OTAApiZmq::ota_start_update(std::string package_path)
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_start_update";
    progress_ = 0;

    UpgradeUpdateReq req{};
    req.set_start_with_precheck(false);
    req.set_ecu_mode(0);
    req.set_package_path(package_path);
    std::string reply{};
    int32_t ret = client_update_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        OTA_API_LOG_ERROR << "zmq request ret err.";
        return -2;
    }

    UpgradeUpdateResp resp{};
    resp.ParseFromString(reply);
    if (resp.error_code() > 0) {
        OTA_API_LOG_WARN << "update start resp error: " << resp.error_code() << ", " << resp.error_msg();
        return -1;
    }
    else {
        OTA_API_LOG_INFO << "upgrade starting.";
    }

    return 0;
}

int32_t
OTAApiZmq::GetState(std::string state)
{
    if (state == "NORMAL_IDLE")
    {
        return NORMAL_IDLE;
    }
    else if (state == "OTA_PRE_UPDATE")
    {
        return OTA_PRE_UPDATE;
    }
    else if (state == "OTA_UPDATING")
    {
        return OTA_UPDATING;
    }
    else if (state == "OTA_UPDATED")
    {
        return OTA_UPDATED;
    }
    else if (state == "OTA_ACTIVING")
    {
        return OTA_ACTIVING;
    }
    else if (state == "OTA_ACTIVED")
    {
        return OTA_ACTIVED;
    }
    else if (state == "OTA_UPDATE_FAILED")
    {
        return OTA_UPDATE_FAILED;
    }
    
    return -1;
}
int32_t
OTAApiZmq::ota_get_update_status()
{
    OTA_API_LOG_INFO << "OTAApiZmq::ota_get_update_status";

    int32_t result = -1;
    UpgradeCommonReq req{};
    req.set_platform(0);
    std::string reply{};
    int32_t ret = client_status_->Request(req.SerializeAsString(), reply, 1000);
    if (ret < 0) {
        OTA_API_LOG_ERROR << "zmq request ret err.";
        return -1;
    }

    UpgradeStatusResp resp{};
    resp.ParseFromString(reply);
    result = GetState(resp.update_status());
    if (resp.error_code() > 0) {
        OTA_API_LOG_WARN << "get update status resp error: " << resp.error_code() << ", " << resp.error_msg();
        result = -2;
    }
    OTA_API_LOG_INFO << "get update status: " << resp.update_status();

    return result;
}


}
}
}




