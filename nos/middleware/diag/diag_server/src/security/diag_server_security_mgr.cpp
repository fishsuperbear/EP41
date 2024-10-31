/*
 * Copyright (c) Hozon Technologies Co., Ltd. 2022-2023. All rights reserved.
 * Description: diag_server_security_mgr.cpp is designed for diagnostic security manager.
 */

#include "diag/diag_sa/include/security_algorithm.h"
#include "diag/diag_server/include/security/diag_server_security_mgr.h"
#include "diag/diag_server/include/session/diag_server_session_handler.h"
#include "diag/diag_server/include/session/diag_server_session_mgr.h"
#include "diag/diag_server/include/common/diag_server_config.h"
#include "diag/diag_server/include/common/diag_server_logger.h"

namespace hozon {
namespace netaos {
namespace diag {

std::mutex DiagServerSecurityMgr::mtx_;
DiagServerSecurityMgr* DiagServerSecurityMgr::instance_ = nullptr;
static bool g_timer_flag = false;
static uint16_t g_sa = 0x00;
static uint16_t g_ta = 0x00;

DiagServerSecurityMgr*
DiagServerSecurityMgr::getInstance()
{
    if (nullptr == instance_) {
        std::lock_guard<std::mutex> lck(mtx_);
        if (nullptr == instance_) {
            instance_ = new DiagServerSecurityMgr();
        }
    }

    return instance_;
}

DiagServerSecurityMgr::DiagServerSecurityMgr()
: current_level(DiagSecurityLevelId_Non)
, step_(0x00)
, seed_(0x00000000)
, access_err_count_(0)
, time_fd_(-1)
, time_mgr_(new TimerManager())
{
    DiagServerSessionMgr::getInstance()->RegisterSessionStatusListener([](DiagServerSessionCode session)->void {
        DG_DEBUG << "DiagServerSecurityMgr::RegisterSessionStatusListener session status " << session;
        DiagServerSecurityMgr::getInstance()->SessionStatusChange(session);
    });
}

void
DiagServerSecurityMgr::Init()
{
    DG_DEBUG << "DiagServerSecurityMgr::Init";
    if (nullptr != time_mgr_) {
        time_mgr_->Init();
    }
}

void
DiagServerSecurityMgr::DeInit()
{
    DG_DEBUG << "DiagServerSecurityMgr::DeInit";
    current_level = DiagSecurityLevelId_Non;
    step_ = 0x00;
    seed_ = 0x00000000;
    access_err_count_ = 0;
    g_timer_flag = false;
    if (nullptr != time_mgr_) {
        time_mgr_->DeInit();
        time_mgr_ = nullptr;
    }

    if (nullptr != instance_) {
        delete instance_;
        instance_ = nullptr;
    }
}

void
DiagServerSecurityMgr::SessionStatusChange(DiagServerSessionCode session)
{
    DG_DEBUG << "DiagServerSecurityMgr::SessionStatusChange session status " << session;
    SetCurrentLevel(DiagSecurityLevelId_Non);
    // init data
    step_ = 0x00;
    seed_ = 0x00000000;
    access_err_count_ = 0;
    g_timer_flag = false;
}

void
DiagServerSecurityMgr::SetCurrentLevel(const uint8_t level)
{
    current_level = level;
    DiagServerSessionMgr::getInstance()->GetDiagSessionInfomation().SetSecurityLevel(current_level);
}

uint8_t
DiagServerSecurityMgr::GetCurrentLevel()
{
    return current_level;
}

void
DiagServerSecurityMgr::AnalyzeUdsMessage(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSecurityMgr::AnalyzeUdsMessage sa: " << udsMessage.udsSa << " ta: " << udsMessage.udsTa;
    g_sa = udsMessage.udsSa;
    g_ta = udsMessage.udsTa;
    // 1.Check the data length < 2
    if (udsMessage.udsData.size() < 0x02) {
        DG_ERROR << "DiagServerSecurityMgr | length error,length < 2!";
        // NRC 0x13
        NegativeResponse(DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
        return;
    }

    // uint8_t sid = udsMessage.udsData[0];
    uint8_t sub_func = udsMessage.udsData[1];
    // 2 3.check sub_func and data length
    if (sub_func == 0x03 || sub_func == 0x05 || sub_func == 0x11) {
        if (udsMessage.udsData.size() != 0x02) {
            DG_ERROR << "DiagServerSecurityMgr | length error: "<< udsMessage.udsData.size() <<", expected length: 2";
            // NRC 0x13
            NegativeResponse(DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
            return;
        }
    }
    else if (sub_func == 0x04 || sub_func == 0x06 || sub_func == 0x12) {
        if (udsMessage.udsData.size() != 0x06) {
            DG_ERROR << "DiagServerSecurityMgr | length error: "<< udsMessage.udsData.size() <<", expected length: 6";
            // NRC 0x13
            NegativeResponse(DiagServerNrcErrc::kIncorrectMessageLengthOrInvalidFormat);
            return;
        }
    }
    else {
        DG_ERROR << "DiagServerSecurityMgr | sub function not supported!";
        // NRC 0x12
        NegativeResponse(DiagServerNrcErrc::kSubfunctionNotSupported);
        return;
    }

    // 4.check sequence
    switch (sub_func) {
    case 0x03:
        {
            if (step_ == 0x00 || step_ == 0x03) {
                step_ = 0x03;
            }
            else if (step_ == 0xFF) {
                DG_ERROR << "DiagServerSecurityMgr | request delay is not over!";
                // NRC 0x37
                NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
                return;
            }
            else {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x24
                NegativeResponse(DiagServerNrcErrc::kRequestSequenceError);
                return;
            }
        }
        break;
    case 0x04:
        {
            if (step_ == 0x03) {
                step_ = 0x04;
            }
            else if (step_ == 0x05) {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x31
                NegativeResponse(DiagServerNrcErrc::kRequestOutOfRange);
                return;
            }
            else if (step_ == 0xFF) {
                DG_ERROR << "DiagServerSecurityMgr | request delay is not over!";
                // NRC 0x37
                NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
                return;
            }
            else {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x24
                NegativeResponse(DiagServerNrcErrc::kRequestSequenceError);
                return;
            }
        }
        break;
    case 0x05:
        {
            if (step_ == 0x00 || step_ == 0x05) {
                step_ = 0x05;
            }
            else if (step_ == 0xFF) {
                DG_ERROR << "DiagServerSecurityMgr | request delay is not over!";
                // NRC 0x37
                NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
                return;
            }
            else {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x24
                NegativeResponse(DiagServerNrcErrc::kRequestSequenceError);
                return;
            }
        }
        break;
    case 0x06:
        {
            if (step_ == 0x05) {
                step_ = 0x06;
            }
            else if (step_ == 0x03) {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x31
                NegativeResponse(DiagServerNrcErrc::kRequestOutOfRange);
                return;
            }
            else if (step_ == 0xFF) {
                DG_ERROR << "DiagServerSecurityMgr | request delay is not over!";
                // NRC 0x37
                NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
                return;
            }
            else {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x24
                NegativeResponse(DiagServerNrcErrc::kRequestSequenceError);
                return;
            }
        }
        break;
    case 0x11:
        {
            if (step_ == 0x00 || step_ == 0x11) {
                step_ = 0x11;
            }
            else if (step_ == 0xFF) {
                DG_ERROR << "DiagServerSecurityMgr | request delay is not over!";
                // NRC 0x37
                NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
                return;
            }
            else {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error! step: " << step_;
                // NRC 0x24
                NegativeResponse(DiagServerNrcErrc::kRequestSequenceError);
                return;
            }
        }
        break;
    case 0x12:
        {
            if (step_ == 0x11) {
                step_ = 0x12;
            }
            else if (step_ == 0xFF) {
                DG_ERROR << "DiagServerSecurityMgr | request delay is not over!";
                // NRC 0x37
                NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
                return;
            }
            else {
                DG_ERROR << "DiagServerSecurityMgr | request sequence error!";
                // NRC 0x24
                NegativeResponse(DiagServerNrcErrc::kRequestSequenceError);
                return;
            }
        }
        break;
    default:
        break;
    }

    DealwithSecurityAccessData(udsMessage);
}

void
DiagServerSecurityMgr::DealwithSecurityAccessData(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSecurityMgr::DealwithSecurityAccessData sa: " << udsMessage.udsSa << " ta: " << udsMessage.udsTa;
    DiagServerUdsMessage pResponse;
    // uint8_t sid = udsMessage.udsData[0];
    uint8_t sub_func = udsMessage.udsData[1];
    uint32_t key = 0x00000000;
    uint32_t mask = 0x00000000;
    uint32_t ret = 0x00;
    g_sa = udsMessage.udsSa;
    g_ta = udsMessage.udsTa;

    pResponse.udsTa = udsMessage.udsSa;
    pResponse.udsSa = udsMessage.udsTa;
    pResponse.udsData.emplace_back(DiagServerServiceReplyOpc::DIAG_SERVER_SERVICE_REPLY_OPC_SECURITY_ACCESS);
    pResponse.udsData.emplace_back(sub_func);

    if ((sub_func % 2) != 0) {
        // If security access is passed
        if (GetCurrentLevel() == sub_func) {
            pResponse.udsData.emplace_back(0x00);
            pResponse.udsData.emplace_back(0x00);
            pResponse.udsData.emplace_back(0x00);
            pResponse.udsData.emplace_back(0x00);
            step_ = 0x00;
            PositiveResponse(pResponse);

            return;
        }

        // The delay is not over
        if (g_timer_flag) {
            DG_ERROR << "DiagServerSecurityMgr | timer1 is not over!";
            // NRC 0x37
            NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
            step_ = 0xFF;
            return;
        }

        mask = DiagServerConfig::getInstance()->GetDiagServerSecurityMask(sub_func);
        ret = SecurityAlgorithm::Instance()->RequestSecurityAlgorithm(sub_func, mask);
        seed_ = ret;
        uint8_t data = ret >> 24;
        pResponse.udsData.emplace_back(data);
        data = ret >> 16;
        pResponse.udsData.emplace_back(data);
        data = ret >> 8;
        pResponse.udsData.emplace_back(data);
        data = ret;
        pResponse.udsData.emplace_back(data);
        PositiveResponse(pResponse);
        return;
    }

    // The delay is not over
    if (g_timer_flag) {
        DG_ERROR << "DiagServerSecurityMgr | timer2 is not over!";
        // NRC 0x37
        NegativeResponse(DiagServerNrcErrc::kRequiredTimeDelayNotExpired);
        step_ = 0xFF;
        return;
    }

    step_ = 0x00;
    key = udsMessage.udsData[2];
    key = (key << 8) | udsMessage.udsData[3];
    key = (key << 8) | udsMessage.udsData[4];
    key = (key << 8) | udsMessage.udsData[5];

    mask = DiagServerConfig::getInstance()->GetDiagServerSecurityMask(sub_func - 1);
    ret = SecurityAlgorithm::Instance()->RequestSecurityAlgorithm(sub_func, mask, seed_);
    if (ret != key) {
        access_err_count_++;
        if (access_err_count_ == 3) {
            DG_ERROR << "DiagServerSecurityMgr | maximum number of attempts exceeded!";
            g_timer_flag = true;
            SetCurrentLevel(DiagSecurityLevelId_Non);
            // NRC 0x36
            NegativeResponse(DiagServerNrcErrc::kExceedNumberOfAttempts);
            StartTimer();
            DG_ERROR << "DiagServerSecurityMgr | maximum number of attempts exceeded!";
        }
        else {
            DG_ERROR << "DiagServerSecurityMgr | Invalid key!";
            SetCurrentLevel(DiagSecurityLevelId_Non);
            // NRC 0x35
            NegativeResponse(DiagServerNrcErrc::kInvalidKey);
        }

        return;
    }

    SetCurrentLevel(static_cast<DiagSecurityLevelId>(sub_func - 1));
    PositiveResponse(pResponse);
}

void
DiagServerSecurityMgr::StartTimer()
{
    DG_DEBUG << "DiagServerSecurityMgr::StartTimer";
    if (time_mgr_ == nullptr) {
        DG_ERROR << "DiagServerSecurityMgr::StartTimer time_mgr_ == null";
        return;
    }

    time_mgr_->StartFdTimer(time_fd_, 10000, std::bind(&DiagServerSecurityMgr::Timeout, this, std::placeholders::_1), NULL, false);
}

void
DiagServerSecurityMgr::StopTimer()
{
    DG_DEBUG << "DiagServerSecurityMgr::StopTimer";
    if (time_mgr_ == nullptr) {
        DG_ERROR << "DiagServerSecurityMgr::StopTimer time_mgr_ == null";
        return;
    }

    time_mgr_->StopFdTimer(time_fd_);
    time_fd_ = -1;
    g_timer_flag = false;
}

void
DiagServerSecurityMgr::Timeout(void * data)
{
    DG_DEBUG << "DiagServerSecurityMgr::Timeout";
    StopTimer();
    access_err_count_ = 0;
    step_ = 0x00;
}

void
DiagServerSecurityMgr::PositiveResponse(const DiagServerUdsMessage& udsMessage)
{
    DG_DEBUG << "DiagServerSecurityMgr::PositiveResponse27";
    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

void
DiagServerSecurityMgr::NegativeResponse(const DiagServerNrcErrc errorCode)
{
    DG_DEBUG << "DiagServerSecurityMgr::NegativeResponse27";
    DiagServerUdsMessage udsMessage;
    udsMessage.udsTa = g_sa;
    udsMessage.udsSa = g_ta;

    udsMessage.udsData.emplace_back(DiagServerNrcErrc::kNegativeHead);
    udsMessage.udsData.emplace_back(DiagServerServiceRequestOpc::DIAG_SERVER_SERVICE_REQUEST_OPC_SECURITY_ACCESS);
    udsMessage.udsData.emplace_back(errorCode);

    DiagServerSessionHandler::getInstance()->ReplyUdsMessage(udsMessage);
}

}  // namespace diag
}  // namespace netaos
}  // namespace hozon
