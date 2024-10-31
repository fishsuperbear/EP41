/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: lock free queue head file.
 * Create: 2020-06-12
 */
#ifndef VRTF_LOCKFREEQUEUE_H
#define VRTF_LOCKFREEQUEUE_H
#include <sched.h>
#include <vector>
namespace vrtf {
namespace vcc {
namespace utils {
template <typename NodeType>
class LockFreeQueue {
public:
    explicit LockFreeQueue(uint32_t maxSize)
        : queueSize_(0U), queueMaxSize_(0U), writeIndex_(0U), readIndex_(0U), maxReadIndex_(0U)
    {
        queueMaxSize_ = (maxSize == UINT32_MAX ? UINT32_MAX : maxSize + 1U);
        queue_.resize(queueMaxSize_);
    }
    LockFreeQueue() = delete;
    ~LockFreeQueue() = default;
    uint32_t GetSize() const
    {
        return queueSize_;
    }

    bool Push(const NodeType &node) noexcept
    {
        uint32_t curWriteIndex;
        uint32_t curReadIndex;

        do {
            curWriteIndex = writeIndex_;
            curReadIndex = readIndex_;
            if (CalcIndex(curWriteIndex + 1U) == CalcIndex(curReadIndex)) {
                /* the queue is full. */
                return false;
            }
        } while (!__sync_bool_compare_and_swap(&writeIndex_, curWriteIndex, curWriteIndex + 1U));

        queue_[CalcIndex(curWriteIndex)] = node;

        while (!__sync_bool_compare_and_swap(&maxReadIndex_, curWriteIndex, curWriteIndex + 1U)) {
            static_cast<void>(sched_yield());
        }
        __sync_fetch_and_add (&queueSize_, 1U);
        return true;
    }

    bool Pop(NodeType &node)
    {
        uint32_t curReadIndex;
        uint32_t curMaxReadIndex;
        do {
            curReadIndex = readIndex_;
            curMaxReadIndex = maxReadIndex_;
            if (CalcIndex(curReadIndex) == CalcIndex(curMaxReadIndex)) {
                /* queue is empty. */
                return false;
            }
            node = queue_[CalcIndex(curReadIndex)];
            queue_[CalcIndex(curReadIndex)] = typename std::decay<NodeType>::type();
            if (__sync_bool_compare_and_swap(&readIndex_, curReadIndex, curReadIndex + 1U)) {
                __sync_fetch_and_sub(&queueSize_, 1U);
                return true;
            }
        } while (true);
    }

    bool GetFront(NodeType &node)
    {
        uint32_t curReadIndex;
        uint32_t curMaxReadIndex;
        do {
            curReadIndex = readIndex_;
            curMaxReadIndex = maxReadIndex_;
            if (CalcIndex(curReadIndex) == CalcIndex(curMaxReadIndex)) {
                /* queue is empty. */
                return false;
            }
            node = queue_[CalcIndex(curReadIndex)];
            if (__sync_bool_compare_and_swap(&readIndex_, curReadIndex, readIndex_)) {
                return true;
            }
        } while (true);
    }

    bool At(uint32_t pos, NodeType &node)
    {
        uint32_t curReadIndex;
        uint32_t curMaxReadIndex;
        do {
            curReadIndex = readIndex_;
            curMaxReadIndex = maxReadIndex_;
            if (CalcIndex(curReadIndex) == CalcIndex(curMaxReadIndex)) {
                /* queue is empty. */
                return false;
            }
            if (pos >= queueSize_) {
                /* exceed queue size. */
                return false;
            }
            node = queue_[CalcIndex(curReadIndex + pos)];
            if (__sync_bool_compare_and_swap(&readIndex_, curReadIndex, readIndex_)) {
                return true;
            }
        } while (true);
    }

    bool Empty() const
    {
        return (GetSize() == 0);
    }

private:
    inline uint32_t CalcIndex(uint32_t num) const
    {
        return num % queueMaxSize_;
    }
    std::vector<NodeType> queue_;
    uint32_t queueSize_;
    uint32_t queueMaxSize_;
    uint32_t writeIndex_;
    uint32_t readIndex_;
    uint32_t maxReadIndex_;
};
}
}
}
#endif // VRTF_LOCKFREEQUEUE_H
