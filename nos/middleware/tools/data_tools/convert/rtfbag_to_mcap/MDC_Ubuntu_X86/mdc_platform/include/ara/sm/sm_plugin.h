/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
 * Description: class Sm Plugin declaration
 */

#ifndef SM_PLUGIN_H
#define SM_PLUGIN_H

#include <memory>
#include "ara/sm/sm_common.h"

namespace ara {
namespace sm {
class SmPlugin final {
public:
    SmPlugin(const SmPlugin &src) = delete;
    SmPlugin operator =(const SmPlugin &src) = delete;
    SmPlugin(SmPlugin &&src) = delete;
    SmPlugin &operator = (SmPlugin &&rhs) const = delete;
    ~SmPlugin();

    static SmPlugin& GetInstance();

    SmResultCode Init(const ara::core::String& name) const;
    SmResultCode UnInit() const;

    SmResultCode RegisterProcTask(const ara::core::String& topic, const TaskHandler& handler) const;
    SmResultCode UnRegisterProcTask(const ara::core::String& topic) const;
private:
    SmPlugin();

    class Impl;
    std::unique_ptr<Impl> pImpl_;
};
} // sm
} // ara
#endif