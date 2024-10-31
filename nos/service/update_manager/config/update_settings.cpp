
#include "update_manager/config/update_settings.h"

#include "json/json.h"
#include <fstream>
#include <string>
#include "update_manager/log/update_manager_logger.h"

namespace hozon {
namespace netaos {
namespace update {

#define UPDATE_SETTING_FILE_FOR_MDC ("/opt/usr/diag_update/mdc-llvm/conf/update_setting.json")
#define UPDATE_SETTING_FILE_FOR_J5  ("/userdata/diag_update/j5/conf/update_setting.json")
#define UPDATE_SETTING_FILE_ORIN    ("/app/runtime_service/update_manager/conf/update_setting.json")
#define UPDATE_SETTING_FILE_DEFAULT ("/app/runtime_service/update_manager/conf/update_setting.json")

#define UPDATE_UPGRADE_PATH         ("/ota/")
#define UPDATE_WORK_PATH            ("/ota/hz_update/")
#define UPDATE_RECOVERY_PATH        ("/ota/recovery/")

UpdateSettings::UpdateSettings()
    : setting_file_("")
    , zip_file_("")
    , path_for_upgrade_(UPDATE_UPGRADE_PATH)
    , path_for_work_(UPDATE_WORK_PATH)
    , path_for_unzip_("./")
    , path_for_bin_files_("./")
    , path_for_update_tmp_("./")
    , path_for_recovery_(UPDATE_RECOVERY_PATH)
    , um_logic_addr_(0x0F00)
    , use_double_compress_(false)
    , same_version_update_(false)
{
}

UpdateSettings& UpdateSettings::Instance()
{
    static UpdateSettings instance;
    return instance;
}

int32_t UpdateSettings::Init()
{
    UM_INFO << "UpdateSettings::Init.";
    ParserUpdateSettings();
    UM_INFO << "UpdateSettings::Init Done.";
    return 0;
}

int32_t UpdateSettings::Deinit()
{
    UM_INFO << "UpdateSettings::Deinit.";
    UM_INFO << "UpdateSettings::Deinit Done.";
    return 0;
}

int32_t UpdateSettings::ParserUpdateSettings()
{
#ifdef BUILD_FOR_MDC
    setting_file_ = UPDATE_SETTING_FILE_FOR_MDC;
#elif BUILD_FOR_J5
    setting_file_ = UPDATE_SETTING_FILE_FOR_J5;
#elif BUILD_FOR_ORIN
    setting_file_ = UPDATE_SETTING_FILE_ORIN;
#else
    setting_file_ = UPDATE_SETTING_FILE_DEFAULT;
#endif

    UPDATE_LOG_D("ParserUpdateSettings file: %s.", setting_file_.c_str());
    int32_t ret = -1;
    std::ifstream cfgFile(setting_file_.c_str(), std::ios::in);
    if (!cfgFile.good()) {
        UPDATE_LOG_E("update setting file: %s read error!", setting_file_.c_str());
        return ret;
    }
    else {
        std::string strJson((std::istreambuf_iterator<char>(cfgFile)),
                            std::istreambuf_iterator<char>());

        Json::CharReaderBuilder readerBuilder;
        std::unique_ptr<Json::CharReader> const reader(readerBuilder.newCharReader());
        Json::Value  rootValue;
        JSONCPP_STRING errs;

        bool res = reader->parse(strJson.c_str(), strJson.c_str() + strlen(strJson.c_str()), &rootValue, &errs);
        if (res && errs.empty()) {
            path_for_unzip_ = rootValue["path_for_unzip"].asString();
            path_for_bin_files_ = rootValue["path_for_bin_files"].asString();
            path_for_update_tmp_ = rootValue["path_for_update_tmp"].asString();
            if ("" != rootValue["path_for_recovery"].asString()) {
                path_for_recovery_ = rootValue["path_for_recovery"].asString();
            }
            path_for_work_ = rootValue["path_for_work"].asString();
            path_for_upgrade_ = rootValue["path_for_upgrade"].asString();
            if (!rootValue["update_manager_logic_addr"].isNull())
            {
                um_logic_addr_ = static_cast<uint16_t>(std::strtoul(rootValue["update_manager_logic_addr"].asString().c_str(), 0, 0));
            }
            if (!rootValue["use_double_compress"].isNull())
            {
                use_double_compress_ = rootValue["use_double_compress"].asBool();
            }
            if (!rootValue["same_version_update"].isNull())
            {
                same_version_update_ = rootValue["same_version_update"].asBool();
            }

            UPDATE_LOG_D("PathForUpgrade: %s", path_for_upgrade_.c_str());
            UPDATE_LOG_D("PathForWork: %s", path_for_work_.c_str());
            UPDATE_LOG_D("PathForUnzip: %s", path_for_unzip_.c_str());
            UPDATE_LOG_D("PathForBinFiles: %s", path_for_bin_files_.c_str());
            UPDATE_LOG_D("PathForUpdateTmp: %s", path_for_update_tmp_.c_str());
            UPDATE_LOG_D("PathForRecovery: %s", path_for_recovery_.c_str());
            UPDATE_LOG_D("setting_file: %s", setting_file_.c_str());
            UPDATE_LOG_D("update_manager_logic_addr: %X", um_logic_addr_);
            UPDATE_LOG_D("UseDoubleCompress: %d", use_double_compress_);
            UPDATE_LOG_D("same_version_update_: %d", same_version_update_);

            ret = 1;
        }
    }
    return ret;
}

std::string UpdateSettings::SettingFile()
{
    return setting_file_;
}

void UpdateSettings::UpdateZipFile(const std::string& file)
{
    zip_file_ = file;
}

std::string UpdateSettings::ZipFile()
{
    return zip_file_;
}

std::string UpdateSettings::PathForUnzip()
{
    return path_for_unzip_;
}

std::string UpdateSettings::PathForBinFiles()
{
    return path_for_bin_files_;
}

std::string UpdateSettings::PathForUpdateTmp()
{
    return path_for_update_tmp_;
}

std::string UpdateSettings::PathForRecovery()
{
    return path_for_recovery_;
}

std::string UpdateSettings::PathForUpgrade()
{
    return path_for_upgrade_;
}

std::string UpdateSettings::PathForWork()
{
    return path_for_work_;
}

std::uint16_t UpdateSettings::UmLogicAddr()
{
    return um_logic_addr_;
}

bool UpdateSettings::UseDoubleCompress()
{
    return use_double_compress_;
}

bool UpdateSettings::SameVersionUpdate()
{
    return same_version_update_;
}

bool UpdateSettings::SetSameVersionUpdate(bool flag)
{
    same_version_update_ = flag;
    return true;
}


}  // namespace update
}  // namespace netaos
}  // namespace hozon