/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: This is used to help register callback about diagnosis.
 * Create: 2021-04-26
 */

#ifndef RTF_COM_DIAGNOSIS_HANDLER_H
#define RTF_COM_DIAGNOSIS_HANDLER_H

#include <memory>
#include <mutex>

#include <someip/SomeipInterface.h>
#include "vrtf/driver/someip/someip_driver_types.h"

namespace vrtf {
namespace driver {
namespace someip {
class DiagnosisHandler : public std::enable_shared_from_this<DiagnosisHandler> {
public:
    static std::shared_ptr<DiagnosisHandler>& GetInstance();
    DiagnosisHandler(DiagnosisHandler &&) = delete;
    DiagnosisHandler(DiagnosisHandler const&) = delete;
    DiagnosisHandler& operator=(DiagnosisHandler const&) = delete;
    DiagnosisHandler& operator=(DiagnosisHandler &&) = delete;
    DiagnosisHandler();
    virtual ~DiagnosisHandler();

    // used by rtf com
    void RegisterFaultsDiagnosisReportCallback(FaultsDiagnosisHandler const &handler) noexcept;
    // Used by driver in initialize someip app
    void RegisterFaultsDiagnosisReportCallback() noexcept;
    void UnregisterFaultsDiagnosisReportCallback(FaultsDiagnosisCallbackType const callbackType) noexcept;
    void ResetDiagnosisCounterReport(ResetDiagnosisCounterType const counterType) noexcept;
    void SetSomeipApplication(const std::shared_ptr<Someip::SomeipInterface>& app) noexcept;
    const std::shared_ptr<Someip::SomeipInterface>& GetSomeipApplication() const noexcept;
    bool IsFaultsDiagnosisHandlerSet() const noexcept;
private:
    void SetCallbacksToSomeip() noexcept;

    bool isCallbackSet_ = false;
    bool isCounterCallbackRegistered_ = false;
    bool isFualtsCallbackRegistered_ = false;
    FaultsDiagnosisHandler faultsDiagnosisHandler_ = {nullptr, nullptr};
    std::shared_ptr<Someip::SomeipInterface> app_ = nullptr;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
    std::mutex registerMutex_;
    std::map<ResetDiagnosisCounterType, Someip::ResetDiagnosisCounterType> const resetMap_ = {
        {ResetDiagnosisCounterType::INVALID_PROTOCOL_VERSION,
         Someip::ResetDiagnosisCounterType::RESET_INVALID_PROTOCOL_VERSION},
        {ResetDiagnosisCounterType::INVALID_INTERFACE_VERSION,
         Someip::ResetDiagnosisCounterType::RESET_INVALID_INTERFACE_VERSION},
        {ResetDiagnosisCounterType::INVALID_SD_MESSAGES,
         Someip::ResetDiagnosisCounterType::RESET_INVALID_SD_MESSAGES},
        {ResetDiagnosisCounterType::INVALID_SD_SUBSCRIBE,
         Someip::ResetDiagnosisCounterType::RESET_INVALID_SD_MESSAGES},
        {ResetDiagnosisCounterType::SERVICE_INDENTIFICATION,
         Someip::ResetDiagnosisCounterType::RESET_SERVICE_IDENTIFICATION},
        {ResetDiagnosisCounterType::INVALID_SERVICE_ID, Someip::ResetDiagnosisCounterType::RESET_INVALID_SERVICE_ID},
        {ResetDiagnosisCounterType::INVALID_METHOD_ID, Someip::ResetDiagnosisCounterType::RESET_INVALID_METHOD_ID},
        {ResetDiagnosisCounterType::INVALID_MESSAGES, Someip::ResetDiagnosisCounterType::RESET_INVALID_MESSAGES},
        {ResetDiagnosisCounterType::ALL, Someip::ResetDiagnosisCounterType::ALL}
    };
};
}
}
}
#endif // RTF_COM_DIAGNOSIS_HELPER_H
