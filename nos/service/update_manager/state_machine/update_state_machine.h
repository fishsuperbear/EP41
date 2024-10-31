#pragma once

#include <stdint.h>
#include <mutex>
#include <map>
#include <set>
#include <memory>

namespace hozon {
namespace netaos {
namespace update {

/*
* 	NORMAL_IDLE        (切换OTA升级模式之前，或者退出OTA升级模式后)
* 	OTA_PRE_UPDATE     (执行升级前的前序动作，包括下载，前编程检查等)
*   OTA_UPDATING       (所有配置需要升级的件升级未完成，且未出错的状态)
*   OTA_UPDATED        (所有配置需要升级的件升级成功)
*   OTA_ACTIVING       (激活过程中，未完成且未发生错误的状态)
*   OTA_ACTIVED        (激活成功)
*   OTA_UPDATE_FAILED  (升级过程或者激活过程发生错误时)
*/
enum class State
{
    NORMAL_IDLE             = 0x01,
    OTA_PRE_UPDATE          = 0x02,
    OTA_UPDATING            = 0x03,
    OTA_UPDATED             = 0x04,
    OTA_ACTIVING            = 0x05,
    OTA_ACTIVED             = 0x06,
    OTA_UPDATE_FAILED       = 0x07,
};

/*
* 	PARSE_CONFIG_FAILED         解析升级包中XML失败
* 	VERIFY_FAILED               验签，或者Hash失败
*   CONFIG_DELETED              升级完成，配置文件被删除
*   ECU_MODE_INVALIED           CMD升级，入参非法
*   SENSOR_UPDATE_FAILED        单独升级Sensor时出现失败
*   SENSOR_NOT_EXIST            CMD升级Sensor但是升级包并没有该sensor
*   SOC_UPDATE_FAILED           Soc升级失败
*   MCU_UPDATE_FAILED           Mcu升级失败
*/
enum class FailedCode
{
    DEFAULT                 = 0x00,
    PARSE_CONFIG_FAILED     = 0x10,
    VERIFY_FAILED           = 0x11,
    CONFIG_DELETED          = 0x12,
    ECU_MODE_INVALIED       = 0x13,
    SENSOR_UPDATE_FAILED    = 0x14,
    SENSOR_NOT_EXIST        = 0x15,
    SOC_UPDATE_FAILED       = 0x16,
    MCU_UPDATE_FAILED       = 0x17,
};

class UpdateStateMachine {
public:

    static UpdateStateMachine* Instance();

    void InitStateMap();
    void Deinit();
    bool SwitchState(State newState, const FailedCode& code = FailedCode::DEFAULT);
    std::string GetCurrentState() const;
    std::string GetPreState() const;
    void SetInitialState(State initialState);
    void ForceSetState(State state);
    uint16_t GetFailedCode();
    std::string GetFailedCodeMsg();
    void SetFailedCode(const FailedCode& code);
    void PostUpdateProcess();
private:
    void AddAllowedTransition(State fromState, State toState);
    std::string GetStateString(State state) const;

private:
    UpdateStateMachine();
    ~UpdateStateMachine();
    UpdateStateMachine(const UpdateStateMachine &);
    UpdateStateMachine & operator = (const UpdateStateMachine &);

    static std::mutex m_mtx;
    static UpdateStateMachine* m_pInstance;

    State currentState;
    State preState;
    std::map<State, std::set<State>> allowedTransitions;
    FailedCode errorCode_{FailedCode::DEFAULT};
};

}  // namespace update
}  // namespace netaos
}  // namespace hozon
