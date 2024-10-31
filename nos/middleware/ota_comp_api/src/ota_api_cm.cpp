/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2023-2023. All rights reserved.
 * Description: ota api definition
 */
#include "ota_api_cm.h"
#include "ota_api_logger.h"

namespace hozon {
namespace netaos {
namespace otaapi {


OTAApi::OTAApi() : progress_(0)
{
    req_data_type_status = std::make_shared<common_reqPubSubType>();
    resp_data_type_status = std::make_shared<update_status_respPubSubType>();
    req_data_status = std::make_shared<common_req>();
    resq_data_status = std::make_shared<update_status_resp>();
    client_status = new Client<common_req, update_status_resp>(req_data_type_status, resp_data_type_status);

    req_data_type_version = std::make_shared<common_reqPubSubType>();
    resp_data_type_version = std::make_shared<get_version_respPubSubType>();
    req_data_version = std::make_shared<common_req>();
    resq_data_version = std::make_shared<get_version_resp>();
    client_version = new Client<common_req, get_version_resp>(req_data_type_version, resp_data_type_version);

    req_data_type_precheck = std::make_shared<common_reqPubSubType>();
    resp_data_type_precheck = std::make_shared<precheck_respPubSubType>();
    req_data_precheck = std::make_shared<common_req>();
    resq_data_precheck = std::make_shared<precheck_resp>();
    client_precheck = new Client<common_req, precheck_resp>(req_data_type_precheck, resp_data_type_precheck);

    req_data_type_progress = std::make_shared<common_reqPubSubType>();
    resp_data_type_progress = std::make_shared<progress_respPubSubType>();
    req_data_progress = std::make_shared<common_req>();
    resq_data_progress = std::make_shared<progress_resp>();
    client_progress = new Client<common_req, progress_resp>(req_data_type_progress, resp_data_type_progress);

    req_data_type_update = std::make_shared<start_update_reqPubSubType>();
    resp_data_type_update = std::make_shared<start_update_respPubSubType>();
    req_data_update = std::make_shared<start_update_req>();
    resq_data_update = std::make_shared<start_update_resp>();
    client_update = new Client<start_update_req, start_update_resp>(req_data_type_update, resp_data_type_update);
}

OTAApi::~OTAApi()
{
}

void
OTAApi::ota_api_init()
{
    OTA_API_LOG_INFO << "OTAApi::ota_api_init";
    client_status->Init(0, "devm_um_1");
    client_version->Init(0, "devm_um_5");
    client_precheck->Init(0, "devm_um_4");
    client_progress->Init(0, "devm_um_3");
    client_update->Init(0, "devm_um_2");
}

void
OTAApi::ota_api_deinit()
{
    OTA_API_LOG_INFO << "OTAApi::ota_api_deinit";
    client_status->Deinit();
    client_version->Deinit();
    client_precheck->Deinit();
    client_progress->Deinit();
    client_update->Deinit();
}

std::string
OTAApi::ota_get_version()
{
    OTA_API_LOG_INFO << "OTAApi::ota_get_api_version";

    std::string version;
    int32_t online = client_version->WaitServiceOnline(3000);  //用户需要去调等待服务
    if (online < 0) {
        OTA_API_LOG_WARN << "WaitServiceOnline err.";
        version = "";
    }
    req_data_version->platform(0);
    client_version->Request(req_data_version, resq_data_version, 2000);


    if (resq_data_version->error_code() > 0) {
        OTA_API_LOG_WARN << "get api version resp error: " << resq_data_version->error_code() << ", " << resq_data_version->error_msg();
        version = "";
    }
    else {
        OTA_API_LOG_INFO << "major_version: " << resq_data_version->major_version();
        OTA_API_LOG_INFO << "soc_version: " << resq_data_version->soc_version();
        OTA_API_LOG_INFO << "mcu_version: " << resq_data_version->mcu_version();
        OTA_API_LOG_INFO << "sensor_version: ";
        for (const auto& pair : resq_data_version->sensor_version()) {
            OTA_API_LOG_INFO << "    " << pair.first << ": " << pair.second;
        }
        version = resq_data_version->major_version();
    }

    return version;
}

int32_t
OTAApi::ota_precheck()
{
    OTA_API_LOG_INFO << "OTAApi::ota_precheck";

    int32_t online = client_precheck->WaitServiceOnline(3000);  //用户需要去调等待服务
    if (online < 0) {
        OTA_API_LOG_WARN << "WaitServiceOnline err.";
        return -1;
    }
    req_data_precheck->platform(0);
    client_precheck->Request(req_data_precheck, resq_data_precheck, 2000);

    if (resq_data_precheck->error_code() > 0) {
        OTA_API_LOG_WARN << "precheck resp error: " << resq_data_precheck->error_code() << ", " << resq_data_precheck->error_msg();
        return -2;
    }
    else {
        if (resq_data_precheck->space() == true 
            && resq_data_precheck->speed() == true 
            && resq_data_precheck->gear() == true) {
            OTA_API_LOG_INFO << "precheck resp true.";
        }
        else {
            OTA_API_LOG_WARN << "precheck resp false," \
                << " space:" << resq_data_precheck->space() \
                << " speed:" << resq_data_precheck->speed() \
                << " gear:" << resq_data_precheck->gear();
            return -3;
        }
    }

    return 0;
}
uint8_t
OTAApi::ota_progress()
{
    OTA_API_LOG_INFO << "OTAApi::ota_progress";

    int32_t online = client_progress->WaitServiceOnline(3000);  //用户需要去调等待服务
    if (online < 0) {
        OTA_API_LOG_WARN << "WaitServiceOnline err.";
        return progress_;
    }
    req_data_progress->platform(0);
    client_progress->Request(req_data_progress, resq_data_progress, 2000);

    if (resq_data_progress->error_code() > 0) {
        OTA_API_LOG_WARN << "progress resp error: " << resq_data_progress->error_code() << ", " << resq_data_progress->error_msg();
        return progress_;
    }
    else {
        OTA_API_LOG_INFO << "progress resp upgrade progress: " << resq_data_progress->progress() << "%";
    }

    progress_ = resq_data_progress->progress();

    return progress_;
}
int32_t
OTAApi::ota_start_update(std::string package_path)
{
    OTA_API_LOG_INFO << "OTAApi::ota_start_update";
    progress_ = 0;

    int32_t online = client_update->WaitServiceOnline(3000);  //用户需要去调等待服务
    if (online < 0) {
        OTA_API_LOG_WARN << "WaitServiceOnline err.";
        return -1;
    }
    req_data_update->start_with_precheck(false);
    req_data_update->package_path(package_path);
    client_update->Request(req_data_update, resq_data_update, 2000);

    if (resq_data_update->error_code() > 0) {
        OTA_API_LOG_WARN << "update start resp error: " << resq_data_update->error_code() << ", " << resq_data_update->error_msg();
        return -2;
    }
    else {
        OTA_API_LOG_INFO << "upgrade starting.";
    }

    return 0;
}

int32_t
OTAApi::GetState(std::string state)
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
OTAApi::ota_get_update_status()
{
    OTA_API_LOG_INFO << "OTAApi::ota_get_update_status";

    int32_t ret = -1;
    int32_t online = client_status->WaitServiceOnline(3000);  //用户需要去调等待服务
    if (online < 0) {
        OTA_API_LOG_WARN << "WaitServiceOnline err.";
        return -1;
    }
    req_data_status->platform(0);
    client_status->Request(req_data_status, resq_data_status, 2000);

    ret = GetState(resq_data_status->update_status());
    if (resq_data_status->error_code() > 0) {
        OTA_API_LOG_WARN << "get update status resp error: " << resq_data_status->error_code() << ", " << resq_data_status->error_msg();
        ret = -2;
    }
    OTA_API_LOG_INFO << "get update status: " << resq_data_status->update_status();

    return ret;
}


}
}
}




