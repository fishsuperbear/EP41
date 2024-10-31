/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: Extension.hpp
 */

#ifndef DDS_CORE_POLICY_EXTENSION_HPP
#define DDS_CORE_POLICY_EXTENSION_HPP

#include <cstdint>

namespace dds {
namespace core {
namespace policy {
/**
 * @brief Configures Qos Extension
 * bandwidth_ is the max bandwidth in 1s
 * sendwindow_ is in ms
 */
class Extension {
public:
    Extension(void) = default;
    ~Extension(void) = default;

    void Bandwidth(uint32_t bandwidth) noexcept
    {
        bandwidth_ = bandwidth;
    }

    void SendWindow(uint32_t sendwindow) noexcept
    {
        sendwindow_ = sendwindow;
    }

    uint32_t Bandwidth(void) const noexcept
    {
        return bandwidth_;
    }

    uint32_t SendWindow(void) const noexcept
    {
        return sendwindow_;
    }

private:
    uint32_t bandwidth_{0U};
    uint32_t sendwindow_{8U};
};
}
}
}

#endif /* DDS_CORE_POLICY_EXTENSION_HPP */
