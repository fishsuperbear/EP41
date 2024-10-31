/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: ThreadGroup head file
 * Create: 2022-06-01
 */
#ifndef ARA_COM_THREAD_GROUP_H
#define ARA_COM_THREAD_GROUP_H

#include <cstdint>

namespace ara {
namespace com {
class ThreadGroup {
public:
    virtual ~ThreadGroup() = default;
    /**
     * @brief Get thread number of the thread group
     * @return thread number
     */
    virtual std::uint16_t GetThreadNumber() const noexcept = 0;
    /**
     * @brief Get queue size of the thread group
     * @return queue size
     */
    virtual std::uint16_t GetQueueSize() const noexcept = 0;
protected:
    ThreadGroup() = default;
};
}
}
#endif  // ARA_COM_THREAD_GROUP_H
