/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2019-2022. All rights reserved.
 * Description: This use to help register dlopen librtf_cm.so to cm.
 * Create: 2022-06-13
 */
#ifndef RTFCM_REGISTER_DFX_INTERFACE_H
#define RTFCM_REGISTER_DFX_INTERFACE_H
#include "rtf/rtfcm/dfx_config_impl.h"
#include "rtf/rtfcm/rtf_maintaind_impl.h"
namespace rtf {
namespace rtfcm {
class RegisterDfxInterface {
public:
    static std::shared_ptr<RegisterDfxInterface>& GetInstance() noexcept;
    RegisterDfxInterface();
    ~RegisterDfxInterface() = default;
    static bool RegisterDfxConfigInterface(const std::shared_ptr<config::DDSDfxConfigInterface> dfxConfigInstance);
    static bool RegisterRtfMaintaindInterface(
        const std::shared_ptr<rtfmaintaind::RtfMaintaindInterface> rtfMntdInstance);
    std::shared_ptr<config::DDSDfxConfigInterface>& GetDfxConfigInstance() const noexcept {
        return dfxConfigInstance_;
    }
    std::shared_ptr<rtfmaintaind::RtfMaintaindInterface>& GetRtfMaintaindInstance() const noexcept {
        return rtfMntdInstance_;
    }
private:
    static std::shared_ptr<config::DDSDfxConfigInterface> dfxConfigInstance_;
    static std::shared_ptr<rtfmaintaind::RtfMaintaindInterface> rtfMntdInstance_;
    std::shared_ptr<ara::godel::common::log::Log> logInstance_;
};
}
}
#endif
