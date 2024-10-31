/*
 * Copyright (c) Hozon Auto Co., Ltd. 2023-2025. All rights reserved.
 * Module: em
 * Description: 配置管理接口
 * Created on: Feb 7, 2023
 * Author: liguoqiang
 *
 */
#include "include/config_param.h"

#include <ctime>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <typeinfo>

#include "src/config_param_impl.h"
#include "src/config_param_impl_proto.h"
namespace hozon {
namespace netaos {
namespace cfg {
using namespace std;
using namespace hozon::netaos::cm;
ConfigParam::ConfigParam() : _pimpl(new ConfigParamImplProto()) {}
// ConfigParam::ConfigParam() : _pimpl(new ConfigParamImpl()) {}
ConfigParam::~ConfigParam() {}
ConfigParam* ConfigParam::Instance() {
    static ConfigParam instance;
    return &instance;
}
CfgResultCode ConfigParam::Init(const uint32_t maxWaitMillis) { return _pimpl->Init(maxWaitMillis); }
template <class T>
CfgResultCode ConfigParam::SetBaseParam(const string& key, const T& value, const ConfigPersistType persistType) {
    return _pimpl->SetParam(key, value, persistType);
}
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const bool& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const uint8_t& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const int32_t& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const float& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const double& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const int64_t& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const string& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<bool>& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<uint8_t>& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<int32_t>& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<float>& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<double>& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<int64_t>& value, const ConfigPersistType persistType);
template CfgResultCode ConfigParam::SetBaseParam(const string& key, const vector<string>& value, const ConfigPersistType persistType);
template <class T>
CfgResultCode ConfigParam::GetBaseParam(const string& key, T& value) {
    return _pimpl->GetParam(key, value);
}
template CfgResultCode ConfigParam::GetBaseParam(const string& key, bool& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, uint8_t& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, int32_t& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, float& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, double& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, int64_t& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, string& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<bool>& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<uint8_t>& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<int32_t>& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<float>& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<double>& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<int64_t>& value);
template CfgResultCode ConfigParam::GetBaseParam(const string& key, vector<string>& value);
template <class T>
CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const T& value) {
    return _pimpl->SetDefaultParam(key, value);
}
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const bool& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const uint8_t& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const int32_t& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const float& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const double& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const int64_t& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const string& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<bool>& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<uint8_t>& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<int32_t>& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<float>& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<double>& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<int64_t>& value);
template CfgResultCode ConfigParam::SetBaseDefaultParam(const string& key, const vector<string>& value);
CfgResultCode ConfigParam::ResetParam(const string& key) { return _pimpl->ResetParam(key); }
template <class T>
CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const T&)> func) {
    return _pimpl->MonitorParam(key, func);
}
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const bool&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const uint8_t&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const int32_t&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const float&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const double&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const int64_t&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const string&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<bool>&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<uint8_t>&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<int32_t>&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<float>&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<double>&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<int64_t>&)> func);
template CfgResultCode ConfigParam::MonitorBaseParam(const string& key, const function<void(const string&, const string&, const vector<string>&)> func);
template <class T>
CfgResultCode ConfigParam::RequestBaseParam(const string& key, const T& value, const uint32_t waitTime) {
    return _pimpl->RequestParam(key, value, waitTime);
}
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const bool& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const uint8_t& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const int32_t& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const float& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const double& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const int64_t& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const string& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<bool>& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<uint8_t>& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<int32_t>& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<float>& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<double>& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<int64_t>& value, const uint32_t waitTime);
template CfgResultCode ConfigParam::RequestBaseParam(const string& key, const vector<string>& value, const uint32_t waitTime);
template <class T>
CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const T& value)> func) {
    return _pimpl->ResponseParam(key, func);
}
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const bool&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const uint8_t&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const int32_t&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const float&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const double&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const int64_t&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const string&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<bool>&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<uint8_t>&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<int32_t>&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<float>&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<double>&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<int64_t>&)> func);
template CfgResultCode ConfigParam::ResponseBaseParam(const string& key, const function<CfgResultCode(const vector<string>&)> func);
CfgResultCode ConfigParam::UnMonitorParam(const string& key) { return _pimpl->UnMonitorParam(key); }
CfgResultCode ConfigParam::DelParam(const string& key) { return _pimpl->DelParam(key); }
CfgResultCode ConfigParam::GetMonitorClients(const string& key, vector<string>& clients) { return _pimpl->GetMonitorClients(key, clients); }
CfgResultCode ConfigParam::GetParamInfoList(vector<CfgParamInfo>& paraminfolist) { return _pimpl->GetParamInfoList(paraminfolist); }
CfgResultCode ConfigParam::GetClientinfolist(vector<CfgClientInfo>& clientinfolist) { return _pimpl->GetClientinfolist(clientinfolist); }
CfgResultCode ConfigParam::DeInit() { return _pimpl->DeInit(); }

}  // namespace cfg
}  // namespace netaos
}  // namespace hozon
