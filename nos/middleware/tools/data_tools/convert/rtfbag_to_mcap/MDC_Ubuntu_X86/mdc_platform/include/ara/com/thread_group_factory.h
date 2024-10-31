/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.
 * Description: ThreadGroupFactory head file
 * Create: 2022-06-01
 */
#ifndef ARA_COM_THREAD_GROUP_FACTORY_H
#define ARA_COM_THREAD_GROUP_FACTORY_H

#include <memory>
#include <cstdint>
namespace ara {
namespace com {
class ThreadGroup;
class ThreadGroupFactory {
public:
    /**
     * @brief Get ThreadGroupFactory singleton instance
     */
    static std::shared_ptr<ThreadGroupFactory>& GetInstance();
    virtual ~ThreadGroupFactory() = default;
    /**
     * @brief Create a thread group
     * @param[in] threadNumber the thread number
     * @param[in] queueSize    the queue size
     * @return whether create thread group success, nullptr for failed
     */
    virtual std::shared_ptr<ara::com::ThreadGroup> CreateThreadGroup(
        const std::uint16_t threadNumber, const std::uint16_t queueSize) = 0;
    /**
     * @brief Delete a thread group
     * @param[in] threadGroup the thread group want to delete
     * @return whether delete thread group success
     */
    virtual bool DeleteThreadGroup(const std::shared_ptr<ara::com::ThreadGroup>& threadGroup) = 0;

    ThreadGroupFactory(const ThreadGroupFactory& rhs) = delete;
    ThreadGroupFactory(ThreadGroupFactory&& rhs) = delete;
    ThreadGroupFactory& operator=(const ThreadGroupFactory& rhs) = delete;
    ThreadGroupFactory& operator=(ThreadGroupFactory&& rhs) = delete;
protected:
    ThreadGroupFactory() = default;
};
}
}
#endif // ARA_COM_THREAD_GROUP_FACTORY_H
