/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: cfg
 * Description: 配置管理接口
 * Created on: Feb 7, 2023
 *
 */

#ifndef MIDDLEWARE_CFG_INCLUDE_CONFIG_PARAM_H_
#define MIDDLEWARE_CFG_INCLUDE_CONFIG_PARAM_H_
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <shared_mutex>
#include <sstream>
#include <string>
#include <vector>

#include "cfg_data_def.h"
#include "serializer_type.h"
namespace hozon {
namespace netaos {
namespace cfg {
// class ConfigParamImpl;
class ConfigParamImplProto;
class ConfigParam {
 public:
    static ConfigParam* Instance();
    ConfigParam();
    virtual ~ConfigParam();
    /**  CFG模块资源初始化，CM通信建立
    @param[in]  maxWaitMillis 服务发现超时时间(ms)，必须指定超时时间最小是10ms.
    @param[out] none
    @return     返回值
    @warning    none
    @note       该初始化接口仅开启服务发现和建立，不保证服务发现成功。实际调用功能接口时，接口内部会判断服务是否可用。
                在应用程序生命周期内，只需要调用一次。
    */
    CfgResultCode Init(const uint32_t maxWaitMillis = 2000);
    template <class T>
    CfgResultCode SetParam(const std::string& key, const T& value, const ConfigPersistType persistType = CONFIG_NO_PERSIST) {
        return SetParamHelper(key, value, persistType);
    }
    template <class T>
    CfgResultCode GetParam(const std::string& key, T& value) {
        return GetParamHelper(key, value);
    }
    template <class T>
    CfgResultCode SetDefaultParam(const std::string& key, const T& value) {
        return SetDefaultParamHelper(key, value);
    }
    CfgResultCode ResetParam(const std::string& key);
    template <class T>
    CfgResultCode RequestParam(const std::string& key, const T& value, const uint32_t waitTime = 1000) {
        return RequestParamHelper(key, value, waitTime);
    }
    // 0： 参数处理流程成功； 非0： 参数处理流程失败
    template <class T>
    CfgResultCode ResponseParam(const std::string& key, const std::function<CfgResultCode(const T& value)> func) {
        return ResponseParamHelper(key, func);
    }
    template <class T>
    CfgResultCode MonitorParam(const std::string& key, const std::function<void(const std::string&, const std::string&, const T&)> func) {
        return MonitorParamHelper(key, func);
    }
    CfgResultCode UnMonitorParam(const std::string& key);
    CfgResultCode DelParam(const std::string& key);

    CfgResultCode GetMonitorClients(const std::string& key, std::vector<std::string>& clients);
    CfgResultCode GetParamInfoList(std::vector<CfgParamInfo>& paraminfolist);
    CfgResultCode GetClientinfolist(std::vector<CfgClientInfo>& clientinfolist);
    /**  CFG模块资源反初始化，CM通信销毁
    @param[in]  none
    @param[out] none
    @return     返回值
    @warning    none
    @note       在应用程序生命周期内，只需要调用一次。
    */
    CfgResultCode DeInit();

 private:
    template <class T>
    CfgResultCode SetBaseParam(const std::string& key, const T& value, const ConfigPersistType persistType);
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode SetParamHelper(const std::string& key, const T& value, const ConfigPersistType persistType) {
        return SetBaseParam<T>(key, value, persistType);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode SetParamHelper(const std::string& key, const std::vector<T>& value, const ConfigPersistType persistType) {
        return SetBaseParam<std::vector<T>>(key, value, persistType);
    }
    template <class T>
    CfgResultCode GetBaseParam(const std::string& key, T& value);
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode GetParamHelper(const std::string& key, T& value) {
        return GetBaseParam<T>(key, value);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode GetParamHelper(const std::string& key, std::vector<T>& value) {
        return GetBaseParam<std::vector<T>>(key, value);
    }
    template <class T>
    CfgResultCode SetBaseDefaultParam(const std::string& key, const T& value);
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode SetDefaultParamHelper(const std::string& key, const T& value) {
        return SetBaseDefaultParam<T>(key, value);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode SetDefaultParamHelper(const std::string& key, const std::vector<T>& value) {
        return SetBaseDefaultParam<std::vector<T>>(key, value);
    }
    template <class T>
    CfgResultCode MonitorBaseParam(const std::string& key, const std::function<void(const std::string&, const std::string&, const T&)> func);
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode MonitorParamHelper(const std::string& key, const std::function<void(const std::string&, const std::string&, const T&)> func) {
        return MonitorBaseParam<T>(key, func);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode MonitorParamHelper(const std::string& key, const std::function<void(const std::string&, const std::string&, const std::vector<T>&)> func) {
        return MonitorBaseParam<std::vector<T>>(key, func);
    }
    template <class T>
    CfgResultCode RequestBaseParam(const std::string& key, const T& value, const uint32_t waitTime);
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode RequestParamHelper(const std::string& key, const T& value, const uint32_t waitTime) {
        return RequestBaseParam<T>(key, value, waitTime);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode RequestParamHelper(const std::string& key, const std::vector<T>& value, const uint32_t waitTime) {
        return RequestBaseParam<std::vector<T>>(key, value, waitTime);
    }
    template <class T>
    CfgResultCode ResponseBaseParam(const std::string& key, const std::function<CfgResultCode(const T& value)> func);
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode ResponseParamHelper(const std::string& key, const std::function<CfgResultCode(const T& value)> func) {
        return ResponseBaseParam<T>(key, func);
    }
    template <typename T, EnableIfBase<T>* = nullptr>
    CfgResultCode ResponseParamHelper(const std::string& key, const std::function<CfgResultCode(const std::vector<T>& value)> func) {
        return ResponseBaseParam<std::vector<T>>(key, func);
    }
    // std::unique_ptr<ConfigParamImpl> _pimpl;
    std::unique_ptr<ConfigParamImplProto> _pimpl;
};
};  // namespace cfg
}  // namespace netaos
}  // namespace hozon

#endif  // MIDDLEWARE_CFG_INCLUDE_CONFIG_PARAM_H_
