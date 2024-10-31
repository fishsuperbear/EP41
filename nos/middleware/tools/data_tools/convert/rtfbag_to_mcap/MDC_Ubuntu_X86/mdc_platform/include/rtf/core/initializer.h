/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: the declaration of the global initialization and shutdown functions that initialize rep.
 * Create: 2020-03-21
 */

#ifndef RTF_CORE_INITIALIZER_H
#define RTF_CORE_INITIALIZER_H

#include <functional>

#include "ara/core/result.h"

namespace rtf {
namespace core {
// Internal interface!!! Prohibit to be called by Application!!!!
class Initializer {
public:
    static std::shared_ptr<Initializer> GetInstance();
    virtual ~Initializer() = default;
    Initializer() = default;
    virtual ara::core::Result<void> Initialize() = 0;
    virtual ara::core::Result<void> Deinitialize() = 0;
    virtual void RegisterDeinitializeCallback(const std::function<void()> &callback) = 0;
    virtual bool IsInitialized() const = 0;

protected:
    Initializer(const Initializer &) = default;
    Initializer &operator = (const Initializer &) & = default;
};
} // End of namespace core
} // End of namespace rtf

#endif
