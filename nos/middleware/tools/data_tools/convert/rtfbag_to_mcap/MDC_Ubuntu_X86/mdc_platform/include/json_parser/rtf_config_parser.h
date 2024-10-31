/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: RtfConfigParser definition
 * Create: 2021-2-25
 */
#ifndef ARA_RTF_CONFIG_PARSER_H
#define ARA_RTF_CONFIG_PARSER_H

#include <string>
#include <memory>

namespace ara {
namespace godel {
namespace common {
namespace jsonParser {
class Document;
}
namespace rtfConfigParser {
class RtfConfigParser final {
public:
    struct RtfDataStatisticConfig {
        bool enable;
        std::uint32_t period;
    };
    struct RtfShmFileOwner {
        std::string user;
        std::string group;
    };
    struct SnapshotConfig {
        std::uint32_t threshold;
        bool enableSchedTrace;
        bool timeDelay;
        bool cpuUsage;
        bool netStatistics;
        std::string path;
    };

    RtfConfigParser();
    explicit RtfConfigParser(std::string const &configPath);
    ~RtfConfigParser() = default;
    bool IsValid() const noexcept;
    std::string GetRtfConfigAddress() const;
    // module could be EM/PHM/TOOLS
    std::string GetRtfConfigInstanceId(std::string const &module) const;
    RtfDataStatisticConfig GetRtfDataStatisticConfig() const;
    RtfShmFileOwner const GetRtfShmFileOwner() const;
    std::string GetRtfPluginsPath(std::string const &pluginName) const;
    bool CheckDataStatisticConfig() const;
    SnapshotConfig const GetSnapshotConfig() const;
    std::string GetDdsTransportModeQos() const;
private:
    void Init();
    void ParseRtfConfig();
    std::string Nic2Addr(std::string const &nicStr) const;
    bool IsValidNumberFormat(std::string const &str) const noexcept;

    std::string configPath_;
    std::shared_ptr<ara::godel::common::jsonParser::Document> doc_ {nullptr};
    bool isValid_ {false};
};
} // namespace rtfConfigParser
} // namespace common
} // namespace rtf
} // namespace ara
#endif
