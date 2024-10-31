#pragma once

#include <stdint.h>
#include <mutex>
#include <memory>
#include "json/json.h"
#include "update_manager/cmd_line_upgrade/cmd_upgrade_def.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_get_version_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_precheck_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_progress_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_start_update_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_update_status_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_start_finish_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_cur_partition_method_receiver.h"
#include "update_manager/cmd_line_upgrade/transfer/devm_switch_slot_method_receiver.h"

namespace hozon {
namespace netaos {
namespace update {

class CmdUpgradeManager {
public:

    static CmdUpgradeManager* Instance();

    void Init();
    void Deinit();
    
    void UpdateStatusMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<update_status_resp_t>& resp);
    void PreCheckMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<precheck_resp_t>& resp);
    void ProgressMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<progress_resp_t>& resp);
    void StartUpdateMethod(const std::shared_ptr<start_update_req_t>& req, std::shared_ptr<start_update_resp_t>& resp);
    void GetVersionMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<get_version_resp_t>& resp);
    void StartFinishMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<start_finish_resp_t>& resp);
    void PartitionMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<cur_pratition_resp_t>& resp);
    void SwitchSlotMethod(const std::shared_ptr<common_req_t>& req, std::shared_ptr<switch_slot_resp_t>& resp);
    
    bool IsCmdTriggerUpgrade();
    bool SetCmdTriggerUpgradeFlag(const bool flag);
    bool SetEcuMode(std::uint16_t mode);
    std::string GetEcuMode();

private:
    CmdUpgradeManager();
    ~CmdUpgradeManager();
    CmdUpgradeManager(const CmdUpgradeManager &);
    CmdUpgradeManager & operator = (const CmdUpgradeManager &);

private:
    bool GetVersionFormKey(const std::string& filePath, const std::string& key, std::string& version);
    bool GetVersionFormKey(const std::string& filePath, const std::string& key, bool& flag);

    static std::mutex m_mtx;
    static CmdUpgradeManager* m_pInstance;

    std::unique_ptr<DevmGetVersionMethodServer> get_version_receiver_;
    std::unique_ptr<DevmPreCheckMethodServer> pre_check_receiver_;
    std::unique_ptr<DevmProgressMethodServer> progress_receiver_;
    std::unique_ptr<DevmStartUpdateMethodServer> start_update_receiver_;
    std::unique_ptr<DevmUpdateStatusMethodServer> update_status_receiver_;
    std::unique_ptr<DevmStartFinishMethodServer> start_finish_receiver_;
    std::unique_ptr<DevmCurPartitionMethodServer> cur_partition_receiver_;
    std::unique_ptr<DevmSwitchSlotMethodServer> switch_slot_receiver_;

    std::atomic<bool> cmd_upgrade_flag_ {false};
    std::uint16_t ecu_mode_;
    std::shared_ptr<start_update_req_t> updatReq_;
    std::string pkgPath_{};
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
