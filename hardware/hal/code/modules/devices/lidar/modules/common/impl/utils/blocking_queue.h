#ifndef _BLOCKING_QUEUE_H_
#define _BLOCKING_QUEUE_H_

#include <deque>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>

template <typename T>
class BlockingQueue
{
public:
    BlockingQueue(uint size, const std::string &name) : size_(size), name_(name) {}
    BlockingQueue() {}

    void init(uint size, const std::string &name)
    {
        size_ = size;
        name_ = name;
    }

    void put(const T x)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        if (size_ != 0 && queue_.size() > size_)
        {
            // TODO
            // LINFO << "queue" << name_ << " is full, size: " << size_;
            queue_.pop_front();
        }
        queue_.push_back(x);
        not_empty_.notify_all();
    }

    T get(size_t index)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty())
        {
            not_empty_.wait(lock);
        }
        return queue_.at(index);
    }

    T take()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        while (queue_.empty())
        {
            not_empty_.wait(lock);
        }
        const T front = queue_.front();
        queue_.pop_front();
        return front;
    }

    size_t size() const
    {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::deque<T> queue_;
    uint size_ = 0;
    std::string name_;
};

#endif // _BLOCKING_QUEUE_H_
