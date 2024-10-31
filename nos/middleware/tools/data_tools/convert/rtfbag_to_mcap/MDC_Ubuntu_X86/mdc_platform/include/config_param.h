/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: config mgr client 对外接口定义
 * Create: 2020-08-10
 */
#ifndef CONFIG_PARAM_H
#define CONFIG_PARAM_H
#include "config_data_define.h"
namespace mdc {
namespace config {
class ConfigParam {
public:
    ConfigParam() = default;
    ~ConfigParam() = default;
    static ConfigParam& GetInstance()
    {
        static ConfigParam cfgParam;
        return  cfgParam;
    }
    static std::int32_t Init(const String& name);
    static std::int32_t SetParam(const String& key, const std::int32_t value,
        const ConfigPersistType persistType = CONFIG_NO_PERSIST);
    static std::int32_t SetParam(const String& key, const String& value,
        const ConfigPersistType persistType = CONFIG_NO_PERSIST);
    static std::int32_t SetParam(const String& key, const Vector<uint8_t>& value,
        const ConfigPersistType persistType = CONFIG_NO_PERSIST);
    static std::int32_t GetParam(const String& key, std::int32_t& value);
    static std::int32_t GetParam(const String& key, String &value);
    static std::int32_t GetParam(const String& key, Vector<uint8_t> &value);
    static std::int32_t MonitorParam(const String& key,
        const std::function<void(const String&, const String&, const std::int32_t&)> func);
    static std::int32_t MonitorParam(const String& key,
        const std::function<void(const String&, const String&, const String&)> func);
    static std::int32_t UnMonitorParam(const String& key);
    static std::int32_t DelParam(const String& key);
    static std::int32_t RequestParam(const String& key, const std::uint32_t reqVal, const std::uint32_t waitTime);
    static std::int32_t RequestParam(const String& key, const String& reqVal, const std::uint32_t waitTime);
    static std::int32_t ResponseParam(const String& key, const std::function<std::int32_t(const std::uint32_t value)> cb);
    static std::int32_t ResponseParam(const String& key, const std::function<std::int32_t(const String& value)> cb);
};
}
}

#endif