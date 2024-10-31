#pragma once
#include <iostream>
#include <thread>

namespace hozon {
namespace orin {
namespace proxy {

enum class UPDATE_RESULT {
    OTA_SUCCESS             = 0x00,
    BSL_UNKNOWN_ERROR_TYPE  = 0x01,
    OTA_RECV_TIMEOUT        = 0x02,
    OTA_ADDR_NOT_VALID      = 0x03,
    OTA_OPEN_FAILED         = 0x04,
    OTA_MD5_CHECK_FAILED    = 0x05,
    OTA_SOCK_FAILED         = 0x06,
    OTA_BIND_FAILED         = 0x07,
    BSL_SUCCESS             = 0x08,
    BSL_FORMAT_ERROR        = 0x09,
    BSL_STATUS_IDEL         = 0x0A,
    BSL_STATUS_BUSY         = 0x0B,
    BSL_NOT_ENOUGH_MEMORY   = 0x0C,
    BSL_SIZE_ERROR          = 0x0D,
    BSL_FLASH_ERROR         = 0x0E,
    BSL_COUNT_ERROR         = 0x0F,
    BSL_SN_ERROR            = 0x10,
    BSL_ERASE_ERROR         = 0x11,
    BSL_ADDRESS_ERROR       = 0x12,
    BSL_MODE_ERROR          = 0x13,
    BSL_BLOCK_TYPE_ERROR    = 0x14,
};

enum class UPDATE_STATUS {
    OTA_ST_BEGIN        = 0x01,
    OTA_ST_INSTALLING   = 0x02,
    OTA_ST_FINISH       = 0x03,
    OTA_ST_FAILED       = 0x04,
};

enum class STD_RTYPE_E {
    E_OK        = 0x01,
    E_NOT_OK    = 0x02,
};

enum class OTA_CURRENT_SLOT {
    OTA_CURRENT_SLOT_A       = 0,
    OTA_CURRENT_SLOT_B       = 1,
};

class OrinUpdateProxy {
public:
    OrinUpdateProxy() {}
    ~OrinUpdateProxy() {}

    enum UPDATE_RESULT RR_StartUpdate(const std::string& file_path) {
        updateProcess_ = true;
        std::cout << "wait ..." << std::endl;
        std::cout << "wait ..." << std::endl;
        std::cout << "wait ..." << std::endl;
        std::cout << "wait ..." << std::endl;
        std::cout << "wait ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(50));
        return UPDATE_RESULT::OTA_SUCCESS; 
    }
    enum STD_RTYPE_E RR_GetUpdateStatus(UPDATE_STATUS& ota_update_status, uint8_t& progress) {
        if (!updateProcess_)
        {
            progress = virtualProgress;
            ota_update_status = updateStatus_;
            return STD_RTYPE_E::E_OK;
        }
        updateStatus_ = UPDATE_STATUS::OTA_ST_INSTALLING;
        virtualProgress += 10;        
        if (virtualProgress >= 100)
        {
            virtualProgress = 100;
            updateProcess_ = false;
            updateStatus_ = UPDATE_STATUS::OTA_ST_FINISH;
        }
        progress = virtualProgress;
        ota_update_status = updateStatus_;
        return STD_RTYPE_E::E_OK; 
    }
    std::string RR_GetVersion(){
        return "DSV_VERSION";
    }
    enum STD_RTYPE_E RR_PartitionSwitch() {
        return STD_RTYPE_E::E_OK;
    }
    enum OTA_CURRENT_SLOT RR_GetCurrentPartition() {
        return OTA_CURRENT_SLOT::OTA_CURRENT_SLOT_A;
    }
    enum STD_RTYPE_E RR_RebootSystem() {
        return STD_RTYPE_E::E_OK;
    }

private:
    UPDATE_STATUS updateStatus_{UPDATE_STATUS::OTA_ST_BEGIN};
    uint8_t virtualProgress {0};
    bool updateProcess_{false};

};

}  // namespace proxy
}  // namespace orin
}  // namespace hozon

