#include <signal.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <map>
#include <ratio>
#include <thread>
#include <vector>
#include <syslog.h>

#include "desay/include/pps_update_manager.h"
#include "update_manager/common/um_functions.h"
#include "update_manager/log/update_manager_logger.h"

// ${header}

USING_DEY_NAMESPACE

int32_t m_UpdateState = 0;
int32_t m_Progress = 0;

int32_t res_code_startupdate = 0;
int32_t res_code_updatestatus = 0;
std::string res_code_getversion = "";
int32_t res_code_switchslot = 0;
int32_t res_code_currentslot = 0;
int32_t res_code_reboot = 0;


static DESY::ppscontrol *m_ppsctrl;

// scope timer
class ScopeTimer
{
public:
    // note:std::chrono::high_resolution_clock::now() 的调用可能会导致kernel负载升高，请谨慎使用--2022-12-16
    ScopeTimer() : m_clock(std::chrono::high_resolution_clock::now()) {}
    ~ScopeTimer() = default;
    template <class Duration = std::chrono::milliseconds>
    int64_t escaped() const
    {
        auto now = std::chrono::high_resolution_clock::now();
        int64_t t = std::chrono::duration_cast<Duration>(now - m_clock).count();
        m_clock = now;
        return t;
    }

private:
    mutable std::chrono::high_resolution_clock::time_point m_clock;
};

template <typename FuncType, typename Duration = std::chrono::milliseconds>
int32_t time_measure(FuncType f)
{
    ScopeTimer st;
    f();
    return static_cast<int32_t>(st.escaped<Duration>());
}

// topic cfg
// topicid,topicname[32],bufcnt,issub,ispub,cmdcnt,datasize,independent_thread,priority,msgtype
static DESY::PPS_CFG_EX_STRU g_ppscfg[] = {
    // publish
    {Dsvpps::E_StartUpdateRequest2DSV, "StartUpdateRequest2DSV", 4, 0, 1, 1, sizeof(Dsvpps::ParameterString), 0, 0},
    {Dsvpps::E_GetUpdateStatusRequest2DSV, "GetUpdateStatusRequest2DSV", 4, 0, 1, 1, sizeof(Dsvpps::PlaceHolder), 0, 0},
    {Dsvpps::E_GetVersionRequest2DSV, "GetVersionRequest2DSV", 4, 0, 1, 1, sizeof(Dsvpps::PlaceHolder), 0, 0},
    {Dsvpps::E_PartitionSwitchRequest2DSV, "PartitionSwitchRequest2DSV", 4, 0, 1, 1, sizeof(Dsvpps::PlaceHolder), 0, 0},
    {Dsvpps::E_GetCurrentPartitionRequest2DSV, "GetCurrentPartitionRequest2DSV", 4, 0, 1, 1, sizeof(Dsvpps::PlaceHolder), 0, 0},
    {Dsvpps::E_RebootSystemRequest2DSV, "RebootSystemRequest2DSV", 4, 0, 1, 1, sizeof(Dsvpps::PlaceHolder), 0, 0},
    // ${publish}
    // subscribe
    {Dsvpps::E_StartUpdateResponse2HZ, "StartUpdateResponse2HZ", 4, 1, 0, 1, sizeof(Dsvpps::UPDATE_RESULT), 0, 0},
    {Dsvpps::E_GetUpdateStatusResponse2HZ, "GetUpdateStatusResponse2HZ", 4, 1, 0, 1, sizeof(Dsvpps::UPDATE_STATUS ), 0, 0},
    {Dsvpps::E_GetVersionResponse2HZ, "GetVersionResponse2HZ", 4, 1, 0, 1, sizeof(Dsvpps::ParameterString), 0, 0},
    {Dsvpps::E_PartitionSwitchResponse2HZ, "PartitionSwitchResponse2HZ", 4, 1, 0, 1, sizeof(Dsvpps::STD_RTYPE_E), 0, 0},
    {Dsvpps::E_GetCurrentPartitionResponse2HZ, "GetCurrentPartitionResponse2HZ", 4, 1, 0, 1, sizeof(Dsvpps::OTA_CURRENT_SLOT ), 0, 0},
    {Dsvpps::E_RebootSystemResponse2HZ, "RebootSystemResponse2HZ", 4, 1, 0, 1, sizeof(Dsvpps::STD_RTYPE_E), 0, 0},
    // ${subscribe}
};

// publish data
static Dsvpps::ParameterString *g_payload_StartUpdateRequest2DSV = new Dsvpps::ParameterString();
Dsvpps::ParameterString &getStartUpdateRequest2DSV() {
    return *g_payload_StartUpdateRequest2DSV;
}

static Dsvpps::PlaceHolder *g_payload_GetUpdateStatusRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getGetUpdateStatusRequest2DSV() {
    return *g_payload_GetUpdateStatusRequest2DSV;
}

static Dsvpps::PlaceHolder *g_payload_GetVersionRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getGetVersionRequest2DSV() {
    return *g_payload_GetVersionRequest2DSV;
}

static Dsvpps::PlaceHolder *g_payload_PartitionSwitchRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getPartitionSwitchRequest2DSV() {
    return *g_payload_PartitionSwitchRequest2DSV;
}

static Dsvpps::PlaceHolder *g_payload_GetCurrentPartitionRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getGetCurrentPartitionRequest2DSV() {
    return *g_payload_GetCurrentPartitionRequest2DSV;
}

static Dsvpps::PlaceHolder *g_payload_RebootSystemRequest2DSV = new Dsvpps::PlaceHolder();
Dsvpps::PlaceHolder &getRebootSystemRequest2DSV() {
    return *g_payload_RebootSystemRequest2DSV;
}

// ${publish_data}

// progress name + _ + datatype name
// ${output}

// ${send_data}
// Send StartUpdateRequest2DSV Data
void sendStartUpdateRequest2DSVData() {
    // send request
    int64_t status = m_ppsctrl->send(Dsvpps::E_StartUpdateRequest2DSV, 0, sizeof(Dsvpps::ParameterString), (char *)g_payload_StartUpdateRequest2DSV);
    if (status < 0)
    {
        std::cout << "sendStartUpdateRequest2DSVData send data failed! error_code = " << status << std::endl;
    }
}

// Send GetUpdateStatusRequest2DSV Data
void sendGetUpdateStatusRequest2DSVData() {
    // send request
    int64_t status = m_ppsctrl->send(Dsvpps::E_GetUpdateStatusRequest2DSV, 0, sizeof(Dsvpps::PlaceHolder), (char *)g_payload_GetUpdateStatusRequest2DSV);
    if (status < 0)
    {
        std::cout << "sendGetUpdateStatusRequest2DSVData send data failed! error_code = " << status << std::endl;
    }
}

// Send GetVersionRequest2DSV Data
void sendGetVersionRequest2DSVData() {
    // send request
    int64_t status = m_ppsctrl->send(Dsvpps::E_GetVersionRequest2DSV, 0, sizeof(Dsvpps::PlaceHolder), (char *)g_payload_GetVersionRequest2DSV);
    if (status < 0)
    {
        std::cout << "sendGetVersionRequest2DSVData send data failed! error_code = " << status << std::endl;
    }
}

// Send PartitionSwitchRequest2DSV Data
void sendPartitionSwitchRequest2DSVData() {
    // send request
    int64_t status = m_ppsctrl->send(Dsvpps::E_PartitionSwitchRequest2DSV, 0, sizeof(Dsvpps::PlaceHolder), (char *)g_payload_PartitionSwitchRequest2DSV);
    if (status < 0)
    {
        std::cout << "sendPartitionSwitchRequest2DSVData send data failed! error_code = " << status << std::endl;
    }
}

// Send GetCurrentPartitionRequest2DSV Data
void sendGetCurrentPartitionRequest2DSVData() {
    // send request
    int64_t status = m_ppsctrl->send(Dsvpps::E_GetCurrentPartitionRequest2DSV, 0, sizeof(Dsvpps::PlaceHolder), (char *)g_payload_GetCurrentPartitionRequest2DSV);
    if (status < 0)
    {
        std::cout << "sendGetCurrentPartitionRequest2DSVData send data failed! error_code = " << status << std::endl;
    }
}

// Send RebootSystemRequest2DSV Data
void sendRebootSystemRequest2DSVData() {
    // send request
    int64_t status = m_ppsctrl->send(Dsvpps::E_RebootSystemRequest2DSV, 0, sizeof(Dsvpps::PlaceHolder), (char *)g_payload_RebootSystemRequest2DSV);
    if (status < 0)
    {
        std::cout << "sendRebootSystemRequest2DSVData send data failed! error_code = " << status << std::endl;
    }
}

using namespace hozon::netaos::update;
class Dispatcher : public HalSubInterface
{
public:
    Dispatcher() = default;
    ~Dispatcher() = default;

    // char *_buf = new char[4194304];

    void onHalSubInterface(int topicid, int cmdid, int size, char *payload)
    {

        // ${topic_recv}
        if (topicid == Dsvpps::E_StartUpdateResponse2HZ)
        {
            // TopicID: StartUpdateResponse2HZ
            if (sizeof(Dsvpps::UPDATE_RESULT) != size) return;
            Dsvpps::UPDATE_RESULT update_result{0};
            memcpy(&update_result, payload, size);
            res_code_startupdate = update_result;
            UM_DEBUG << "pps: update_result = " << GetDesayUpdateResultString(res_code_startupdate);
        }
        else if (topicid == Dsvpps::E_GetUpdateStatusResponse2HZ)
        {
            // TopicID: GetUpdateStatusResponse2HZ
            if (sizeof(Dsvpps::UPDATE_STATUS ) != size) return;
            Dsvpps::UPDATE_STATUS update_status = {0};
            memcpy(&update_status, payload, size);
            UM_DEBUG << "pps: update status = " << GetDesayUpdateStatusString(update_status.status) << "; progress = " << update_status.progress;
            m_Progress = update_status.progress;
            m_UpdateState = update_status.status;
        }
        else if (topicid == Dsvpps::E_GetVersionResponse2HZ)
        {
            // TopicID: GetVersionResponse2HZ
            if (sizeof(Dsvpps::ParameterString) != size) return;
            Dsvpps::ParameterString version_result;
            memcpy(&version_result, payload, size);
            res_code_getversion.assign(version_result.begin(), version_result.end());
            UM_DEBUG << "pps: get version is : " << res_code_getversion;
        }
        else if (topicid == Dsvpps::E_PartitionSwitchResponse2HZ)
        {
            // TopicID: PartitionSwitchResponse2HZ
            if (sizeof(Dsvpps::STD_RTYPE_E) != size) return; 
            Dsvpps::STD_RTYPE_E result{};
            memcpy(&result, payload, size);
            res_code_switchslot = static_cast<int32_t>(result);
            UM_DEBUG << "pps: PartitionSwitchResponse = " << GetDesayUpdateString(res_code_switchslot);
        }
        else if (topicid == Dsvpps::E_GetCurrentPartitionResponse2HZ)
        {
            // TopicID: GetCurrentPartitionResponse2HZ
            if (sizeof(Dsvpps::OTA_CURRENT_SLOT ) != size) return;
            Dsvpps::OTA_CURRENT_SLOT result{};
            memcpy(&result, payload, size);
            res_code_currentslot = static_cast<int32_t>(result);
            UM_DEBUG << "pps: GetCurrentPartitionResponse2HZ = " << GetDesayUpdateCurPartitonString(res_code_currentslot);
        }
        else if (topicid == Dsvpps::E_RebootSystemResponse2HZ)
        {
            // TopicID: RebootSystemResponse2HZ
            if (sizeof(Dsvpps::STD_RTYPE_E) != size) return;
            Dsvpps::STD_RTYPE_E result{};
            memcpy(&result, payload, size);
            res_code_reboot = static_cast<int32_t>(result);
            UM_DEBUG << "pps: RebootSystemResponse = " << GetDesayUpdateString(res_code_reboot);
        }
    }
};

void stateCallback(int topicid, int param, int errid, char *errstr)
{
    // std::cout << "stateCallback topicid = " << topicid << " param = " << param << "errid = " << errid << std::endl;
}

void initPPS()
{
    // init pps_com.
    m_ppsctrl = DESY::ppscontrol::Instance(g_ppscfg, sizeof(g_ppscfg) / sizeof(g_ppscfg[0]));
    m_ppsctrl->setstrategy(1, 1500);

    static Dispatcher disp;
    DESY::ppscontrol::Instance()->registerCallback(&disp);
    DESY::ppscontrol::Instance()->registeronstate(&stateCallback);
}

int32_t Get_Code_StartUpdate()
{
    return res_code_startupdate;
}
int32_t Get_Code_SwitchSlot()
{
    return res_code_switchslot;
}
std::string Get_Code_GetCurrentSlot()
{
    return GetDesayUpdateCurPartitonString(res_code_currentslot);
}
std::string Get_Code_GetVersion()
{
    return res_code_getversion;
}
int32_t Get_Code_Reboot()
{
    return res_code_reboot;
}

int32_t Get_UpdateState()
{
    return m_UpdateState;
}

int32_t Get_Update_Progress()
{
    return m_Progress;
}


