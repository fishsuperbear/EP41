#include <thread>
#include <iostream>

#include "sys_statemgr/include/dispatcher.h"
#include "sys_statemgr/include/logger.h"

namespace hozon {
namespace netaos {
namespace ssm {

const uint8_t dsv2hz_msg_body[][5] = {
    {0x17, 0x10, 0x01, 0x00, 0x02},
    {0x17, 0x10, 0x01, 0x00, 0x03}
};

const uint8_t hz2dsv_msg_body[5] = {0x18, 0x10, 0x01, 0x00, 0x02};

const uint16_t u16_crc_table[16] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef};

static DESY::PPS_CFG_EXX_STRU g_ppscfg[] = {
    /* publish */
    {Dsvpps::E_PowerMgrMsg2DSV, "PowerMgrMsg2DSV", 4, 0, 1, 1, sizeof(Dsvpps::DsvPowerMgrMsg_Array), 0, 0, MSG_BLOCKWAIT},
    /* subscribe */
    {Dsvpps::E_PowerMgrMsg2HZ, "PowerMgrMsg2HZ", 4, 100000, 0, 1, sizeof(Dsvpps::DsvPowerMgrMsg_Array), 0, 0, MSG_BLOCKWAIT}
};

Dispatcher::Dispatcher() : m_stopFlag(0) {
}

Dispatcher::~Dispatcher() {}

void Dispatcher::Init(std::shared_ptr<StateManager> ptr) {
    m_smgr = ptr;
    InitPPS();
}

void Dispatcher::InitPPS() {
    m_ppsctl = DESY::ppscontrol::Instance(g_ppscfg, sizeof(g_ppscfg) / sizeof(g_ppscfg[0]));
    m_ppsctl->setstrategy(1, 1500);
    DESY::ppscontrol::Instance()->registerCallback(std::bind(&Dispatcher::onHalSubInterface, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    DESY::ppscontrol::Instance()->registeronstate(&stateCallback);
} 

void Dispatcher::DeInit() {
    m_stopFlag = 1;
    if (m_thr_pm.joinable()) {
        m_thr_pm.join();
    }
}

void Dispatcher::Run() {
    m_thr_pm = std::thread([this]() {
    while (!m_stopFlag) {
		if (!m_que_mstate.empty()) {
            SocModeState mstate = DequeueSocModeState();
            switch (mstate)
            {
            case SocModeState::SOC_MODE_STANDBY:
                break;
            case SocModeState::SOC_MODE_SHUTDOWN: {
                std::string smode = "Standby";
                m_smgr->SwitchMode(smode);
                DsvPowerMgrMsg_Array arry;
                EncodeDsvPowerMgrMsg(&arry);
                SendPowerMgrMsg2DSVData(&arry);
                }
                break;
            default:
                break;
            } 
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
	    }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    });
}

void Dispatcher::onHalSubInterface(int topicid, int cmdid, int size, char *payload) {
    SSM_LOG_INFO << "recv topicid:" << topicid << " cmdid:" << cmdid << " data size:" << size;
    if (topicid == Dsvpps::E_PowerMgrMsg2HZ)
    {
        /* TopicID: PowerMgrMsg2HZ */
        if (sizeof(DsvPowerMgrMsg_Array) != size) {
            SSM_LOG_ERROR << "invalid data size:" << sizeof(DsvPowerMgrMsg_Array);
        } else {
            memset(&m_arry, 0, size);
            memcpy(&m_arry, payload, size);

            std::vector<uint8_t> vec;
            for (size_t i = 0; i < sizeof(m_arry) / sizeof(m_arry[0]); i++) {
                vec.push_back(m_arry[i]);
            }
            SSM_LOG_INFO << UM_UINT8_VEC_TO_HEX_STRING(vec);
            
            PowerMsgHandle(&m_arry);
        }
    }
}

void Dispatcher::stateCallback(int topicid, int param, int errid, char *errstr) {
    SSM_LOG_INFO <<"stateCallback topicid:" << topicid << " param:" << param << "errid:" << errid;
}

void Dispatcher::SendPowerMgrMsg2DSVData(const DsvPowerMgrMsg_Array * arry) {
    /* send to dsv */
    int64_t status = m_ppsctl->send(Dsvpps::E_PowerMgrMsg2DSV, 0, sizeof(DsvPowerMgrMsg_Array), (char *)arry);
    std::vector<uint8_t> vec;
    for (size_t i = 0; i < arry->size(); i++) {
        vec.push_back(arry->at(i));
    }
    SSM_LOG_INFO << UM_UINT8_VEC_TO_HEX_STRING(vec);
    if (status < 0) {
        SSM_LOG_ERROR << "send power msg to dsv failed,ecode:" << status;
    } else {
        SSM_LOG_INFO << "send power msg to dsv succss";
    }
}

void Dispatcher::PowerMsgHandle(DsvPowerMgrMsg_Array *arry) {
    if (arry) {
        SSM_LOG_INFO << "decode dsv mgr msg (pos 4):" << arry->at(2) <<"," << arry->at(3) << "," <<arry->at(4);
        // if(DecodeDsvPowerMgrMsg(arry)) {
            switch (arry->at(4))
            {
            case 0:
            case 1:
                break;
            case 2: /* Start Power-off */
                SSM_LOG_INFO << "recv dsv power-off start";
                EnqueueSocModeState(SocModeState::SOC_MODE_SHUTDOWN);
                break;
            case 3: /* Confirmed Power-off */
                SSM_LOG_INFO << "recv dsv power-off confirm";
                break;
            default:
                SSM_LOG_WARN << "recv dsv unknow msg";
                break;
            }
        // }
    }
}

bool Dispatcher::DecodeDsvPowerMgrMsg(DsvPowerMgrMsg_Array *arry) {
    if (arry) {
        size_t row = sizeof(dsv2hz_msg_body) / sizeof(dsv2hz_msg_body[0]);
        size_t col = sizeof(dsv2hz_msg_body[0]) / sizeof(dsv2hz_msg_body[0][0]);
        for (size_t i = 0; i < row; i++) {
            uint16_t val = CRC16(const_cast<uint8_t *>(dsv2hz_msg_body[i]), col);
            if(arry->at(5) == (val >> 8 & 0xFF) && arry->at(6) == (val & 0xFF)) {
                return true;
            }
        }
    }
    SSM_LOG_ERROR << "crc check failed";
    return false;
}

void Dispatcher::EncodeDsvPowerMgrMsg(DsvPowerMgrMsg_Array *arry) {
    if (arry) {
        memcpy(arry, hz2dsv_msg_body, 5);
        std::get<5>(*arry) = 0x00;
        std::get<6>(*arry) = 0x00;
        // uint16_t val = CRC16(const_cast<uint8_t *>(hz2dsv_msg_body), 5);
        // memcpy(arry, hz2dsv_msg_body, 5);
        // std::get<5>(*arry) = (val >> 8 & 0xFF);
        // std::get<6>(*arry) = (val & 0xFF);
    }
}

void Dispatcher::EnqueueSocModeState(SocModeState state) {
    std::lock_guard<std::mutex> lock(m_mutex_mstate);
    if (m_que_mstate.size() > 10) {
        SSM_LOG_ERROR <<"switch mode que blocked";
    } else {
        m_que_mstate.push(state);
        if (m_que_mstate.size() > 2) {
            SSM_LOG_WARN <<"switch mode que size:"<<m_que_mstate.size();
        }
    }
}
    
SocModeState Dispatcher::DequeueSocModeState() {
    std::lock_guard<std::mutex> lock(m_mutex_mstate);
    SocModeState st = m_que_mstate.front();
    m_que_mstate.pop();
    return st;
}

void Dispatcher::EmptyQueueSocModeState() {
    std::lock_guard<std::mutex> lock(m_mutex_mstate);
    if (!m_que_mstate.empty()) {
        std::queue<SocModeState> empty;
        swap(empty,m_que_mstate);
    }
}


uint16_t Dispatcher::CRC16(uint8_t *data, uint16_t len) {
    uint16_t crc16 = 0xFFFF;
    uint16_t crc_h4, crc_l12;
    while (len--) {
        crc_h4 = (crc16 >> 12);
        crc_l12 = (crc16 << 4);
        crc16 = crc_l12 ^ u16_crc_table[crc_h4 ^ (*data >> 4)];
        crc_h4 = (crc16 >> 12);
        crc_l12 = (crc16 << 4);
        crc16 = crc_l12 ^ u16_crc_table[crc_h4 ^ (*data & 0x0f)];
        data++;
    }
    return crc16;
}


}}}