/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2023. All rights reserved.
 */

#ifndef MDC_DEVM_DEVMC_DEVMCENTERSERVICEINTERFACE_COMMON_H
#define MDC_DEVM_DEVMC_DEVMCENTERSERVICEINTERFACE_COMMON_H

#include "ara/com/types.h"
#include "ara/com/init_config.h"
#include "impl_type_uint8.h"
#include "impl_type_int32_t.h"
#include "impl_type_string.h"
#include "impl_type_uint16.h"
#include "mdc/devm/impl_type_uint8list.h"
#include "impl_type_int32.h"
#include "impl_type_stringlist.h"
#include "mdc/devm/impl_type_deviceinfo.h"
#include "mdc/devm/impl_type_devicelist.h"
#include "impl_type_uint8_t.h"
#include "mdc/devm/impl_type_workstatustype.h"
#include "mdc/devm/impl_type_poweroffinfotype.h"
#include "mdc/devm/impl_type_upgradedevlist.h"
#include "mdc/devm/impl_type_upgradeinfo.h"
#include "mdc/devm/impl_type_sensorinmsg.h"
#include "mdc/devm/impl_type_canctrlconfig.h"
#include "mdc/devm/impl_type_items.h"
#include "impl_type_uint32.h"
#include "mdc/devm/impl_type_cameravideocomplinkinfolist.h"
#include <cfloat>
#include <cmath>

namespace mdc {
namespace devm {
namespace devmc {
namespace methods {
namespace CanNetMgrCtrl {
struct Output {
    ::int32_t result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace CanNetMgrCtrl
namespace DidOperate {
struct Output {
    ::mdc::devm::Uint8List rxData;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(rxData);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(rxData);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (rxData == t.rxData) && (result == t.result);
    }
};
} // namespace DidOperate
namespace DoSensorAction {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace DoSensorAction
namespace DoSystemAction {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace DoSystemAction
namespace GetConfig {
struct Output {
    ::mdc::devm::Uint8List config;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(config);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(config);
    }

    bool operator==(const Output& t) const
    {
       return (config == t.config);
    }
};
} // namespace GetConfig
namespace GetDevAttribute {
struct Output {
    ::String result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace GetDevAttribute
namespace GetDeviceInfo {
struct Output {
    ::mdc::devm::DeviceInfo devInfo;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(devInfo);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(devInfo);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (devInfo == t.devInfo) && (result == t.result);
    }
};
} // namespace GetDeviceInfo
namespace GetDeviceList {
struct Output {
    ::mdc::devm::DeviceList deviceList;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(deviceList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(deviceList);
    }

    bool operator==(const Output& t) const
    {
       return (deviceList == t.deviceList);
    }
};
} // namespace GetDeviceList
namespace GetMasterMode {
struct Output {
    ::UInt8 mode;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(mode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(mode);
    }

    bool operator==(const Output& t) const
    {
       return (mode == t.mode);
    }
};
} // namespace GetMasterMode
namespace GetPowerOptType {
struct Output {
    ::uint8_t optType;
    ::int32_t result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(optType);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(optType);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (optType == t.optType) && (result == t.result);
    }
};
} // namespace GetPowerOptType
namespace GetState {
struct Output {
    ::mdc::devm::WorkStatusType State;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(State);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(State);
    }

    bool operator==(const Output& t) const
    {
       return (State == t.State);
    }
};
} // namespace GetState
namespace GetStatisticsInfo {
struct Output {
    ::String statisticsInfo;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(statisticsInfo);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(statisticsInfo);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (statisticsInfo == t.statisticsInfo) && (result == t.result);
    }
};
} // namespace GetStatisticsInfo
namespace GetSysPowerOffInfo {
struct Output {
    ::mdc::devm::PowerOffInfoType powerOffInfo;
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(powerOffInfo);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(powerOffInfo);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (powerOffInfo == t.powerOffInfo) && (result == t.result);
    }
};
} // namespace GetSysPowerOffInfo
namespace GetTemperature {
struct Output {
    ::String temperature;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(temperature);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(temperature);
    }

    bool operator==(const Output& t) const
    {
       return (temperature == t.temperature);
    }
};
} // namespace GetTemperature
namespace GetUpdateState {
struct Output {
    ::UInt8 progress;
    ::Int32 errorCode;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(progress);
        fun(errorCode);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(progress);
        fun(errorCode);
    }

    bool operator==(const Output& t) const
    {
       return (progress == t.progress) && (errorCode == t.errorCode);
    }
};
} // namespace GetUpdateState
namespace GetUpgradableDeviceList {
struct Output {
    ::mdc::devm::UpgradeDevList upgradeDevList;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(upgradeDevList);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(upgradeDevList);
    }

    bool operator==(const Output& t) const
    {
       return (upgradeDevList == t.upgradeDevList);
    }
};
} // namespace GetUpgradableDeviceList
namespace GetUpgradeInfo {
struct Output {
    ::mdc::devm::UpgradeInfo upgradeInfo;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(upgradeInfo);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(upgradeInfo);
    }

    bool operator==(const Output& t) const
    {
       return (upgradeInfo == t.upgradeInfo);
    }
};
} // namespace GetUpgradeInfo
namespace GetVoltage {
struct Output {
    ::String voltage;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(voltage);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(voltage);
    }

    bool operator==(const Output& t) const
    {
       return (voltage == t.voltage);
    }
};
} // namespace GetVoltage
namespace GetWorkMode {
struct Output {
    ::UInt8 mode;
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(mode);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(mode);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (mode == t.mode) && (result == t.result);
    }
};
} // namespace GetWorkMode
namespace SensorMsgProxy {
struct Output {
    ::mdc::devm::Uint8List respData;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(respData);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(respData);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (respData == t.respData) && (result == t.result);
    }
};
} // namespace SensorMsgProxy
namespace SetCanCtrlConfig {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetCanCtrlConfig
namespace SetConfig {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetConfig
namespace SetMasterMode {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetMasterMode
namespace SetMcuSystemStatus {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetMcuSystemStatus
namespace SetSysPowerOffInfoReaded {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetSysPowerOffInfoReaded
namespace SetWorkMode {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetWorkMode
namespace TranConfig {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace TranConfig
namespace Update {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace Update
namespace UpdateSyncStart {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace UpdateSyncStart
namespace OpenWatchDog {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return true;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace OpenWatchDog
namespace GetEcuInfo {
struct Output {
    ::mdc::devm::Uint8List ecuInfo;
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(ecuInfo);
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(ecuInfo);
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (ecuInfo == t.ecuInfo) && (result == t.result);
    }
};
} // namespace GetEcuInfo
namespace SetCameraVideoCompLinkInfo {
struct Output {
    ::Int32 result;

    static bool IsPlane()
    {
        return false;
    }

    using IsEnumerableTag = void;
    template<typename F>
    void enumerate(F& fun)
    {
        fun(result);
    }

    template<typename F>
    void enumerate(F& fun) const
    {
        fun(result);
    }

    bool operator==(const Output& t) const
    {
       return (result == t.result);
    }
};
} // namespace SetCameraVideoCompLinkInfo
} // namespace methods

class DevmCenterServiceInterface {
public:
    constexpr DevmCenterServiceInterface() = default;
    constexpr static ara::com::ServiceIdentifierType ServiceIdentifier = ara::com::ServiceIdentifierType("/HuaweiMDC/PlatformServiceInterface/DevmCenterServiceInterface/DevmCenterServiceInterface");
    constexpr static ara::com::ServiceVersionType ServiceVersion = ara::com::ServiceVersionType("1.1");
};
} // namespace devmc
} // namespace devm
} // namespace mdc

#endif // MDC_DEVM_DEVMC_DEVMCENTERSERVICEINTERFACE_COMMON_H
