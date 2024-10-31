/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2023. All rights reserved.
 * Description:  Class UpdateCTTaskInterfaceActivate implement
 */
#include "json/json.h"
#include "update_cttask_interface_activate.h"
#include "update_manager/agent/ota_agent.h"
#include "update_manager/taskbase/update_task_event.h"
#include "update_manager/common/data_def.h"
#include "update_manager/record/ota_store.h"
#include "update_manager/config/config_manager.h"
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

UpdateCTTaskInterfaceActivate::UpdateCTTaskInterfaceActivate(STObject* pParent, STObject::TaskCB pfnCallback)
    : CommandTaskBase(OTA_CTTASK_INTERFACE_ACTIVATE, pParent, pfnCallback)
    , curMajorVer_{}, imgMajorVer_{}
{
}

UpdateCTTaskInterfaceActivate::~UpdateCTTaskInterfaceActivate()
{
}

uint32_t 
UpdateCTTaskInterfaceActivate::doCommand()
{
    GetAllImgVersion();
    GetAllCurrentVersion();
    if (waitEvent(OTA_TIMER_P2_CLIENT)) {
        return eContinue;
    }
    return N_ERROR;
}

bool 
UpdateCTTaskInterfaceActivate::onEventAction(bool isTimeout, STEvent* event)
{
    if (isTimeout) {
        auto res = AllVersionCheck();
        if (res) {
            std::string packageName{};
            OTAStore::Instance()->ReadPackageNameData(packageName);

            UPDATE_LOG_D("FileRecovery src: %s, target: %s.", packageName.c_str(), UpdateSettings::Instance().PathForRecovery().c_str());
            FileRecovery(packageName, UpdateSettings::Instance().PathForRecovery());
            setTaskResult(N_OK);
        } else {
            setTaskResult(N_ERROR);
        }
        return true;
    }

    return false;
}

void 
UpdateCTTaskInterfaceActivate::GetAllImgVersion()
{
    auto res = ParseVersionFile();
    if (!res) {
        UM_ERROR << "Parse error !";
    }
    UM_INFO << "app img major version from cfg is : " << imgMajorVer_;
}

void 
UpdateCTTaskInterfaceActivate::GetAllCurrentVersion()
{
    std::vector<uint8_t> version{};
    OTAStore::Instance()->ReadECUVersionData(version);
    curMajorVer_.assign(version.begin(), version.end());
    UM_INFO << "current major version from cfg is : " << curMajorVer_;
}

bool 
UpdateCTTaskInterfaceActivate::AllVersionCheck()
{
    size_t found = imgMajorVer_.find(curMajorVer_);
    if (found == std::string::npos) {
        UM_ERROR << "dsv version compare failed. current dsv Version is : " << curMajorVer_ << ", img version is : " << imgMajorVer_;
        return false;
    }
    return true;
}

bool
UpdateCTTaskInterfaceActivate::ParseVersionFile()
{
    UPDATE_LOG_D("ParseVersionFile");

    std::string versionFile = VERSION_FILE_PATH;
    UPDATE_LOG_D("versionFile: %s", versionFile.c_str());

    if (!PathExists(versionFile)){
        UPDATE_LOG_W("version.json not Exists");
        return false;
    }

    Json::Value rootReder;
    Json::CharReaderBuilder readBuilder;
    std::ifstream ifs(versionFile);
    JSONCPP_STRING errs;
    bool res = Json::parseFromStream(readBuilder, ifs, &rootReder, &errs);
    if(!res) {
        ifs.close();
        UPDATE_LOG_E("Json parse error.");
        return false;
    }
    // Major Version
    if (!rootReder["HZ"].isNull()) {
        imgMajorVer_ = rootReder["HZ"].asString();
    }
    ifs.close();
    return true;
}

} // end of update
} // end of netaos
} // end of hozon
/* EOF */
