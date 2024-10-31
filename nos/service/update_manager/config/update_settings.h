
#ifndef UPDATE_CONFIG_SETTINGS_H_
#define UPDATE_CONFIG_SETTINGS_H_

#include <string>

namespace hozon {
namespace netaos {
namespace update {

class UpdateSettings {
public:
    UpdateSettings();
    static UpdateSettings &Instance();

    int32_t Init();
    int32_t Deinit();

    int32_t ParserUpdateSettings();

    std::string SettingFile();
    void UpdateZipFile(const std::string& file);
    std::string ZipFile();
    std::string PathForUnzip();
    std::string PathForBinFiles();
    std::string PathForUpdateTmp();
    std::string PathForRecovery();
    std::string PathForUpgrade();
    std::string PathForWork();
    std::uint16_t UmLogicAddr();
    bool UseDoubleCompress();
    bool SameVersionUpdate();
    bool SetSameVersionUpdate(bool flag);

private:
    std::string setting_file_;
    std::string zip_file_;
    std::string path_for_upgrade_;
    std::string path_for_work_;
    std::string path_for_unzip_;
    std::string path_for_bin_files_;
    std::string path_for_update_tmp_;
    std::string path_for_recovery_;
    std::uint16_t um_logic_addr_;
    bool use_double_compress_;
    bool same_version_update_;
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
#endif  // UPDATE_CONFIG_SETTINGS_H_