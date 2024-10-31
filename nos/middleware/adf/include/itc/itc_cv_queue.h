#pragma once

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

namespace hozon {
namespace netaos {
namespace adf {

template <typename T>
class SizeLimitQueue {
   public:
    SizeLimitQueue(uint32_t capacity) : _capacity(capacity) {}

    std::shared_ptr<T> ReadLatestOne(bool erase) {
        if (_container.empty()) {
            return nullptr;
        }

        std::shared_ptr<T> data(std::make_shared<T>(_container.back()));
        if (erase) {
            _container.pop_back();
        }

        return data;
    }

    std::shared_ptr<T> ReadOldestOne(bool erase) {
        if (_container.empty()) {
            return nullptr;
        }

        std::shared_ptr<T> res(std::make_shared<T>(_container.front()));
        if (erase) {
            _container.pop_front();
        }

        return res;
    }

    std::vector<std::shared_ptr<T>> ReadLatestN(std::size_t n, bool erase) {
        uint32_t len = std::min(n, _container.size());
        std::vector<std::shared_ptr<T>> out;
        out.reserve(len);
        auto bound = _container.rbegin();
        std::advance(bound, len);
        for (auto it = _container.rbegin(); (it != _container.rend()) && (it != bound); ++it) {
            out.emplace_back(std::make_shared<T>(std::ref(*it)));
        }

        if (erase) {
            for (size_t i = 0; i < len; ++i) {
                _container.pop_back();
            }
        }

        return out;
    }

    void PushOne(const T& val) {
        while (_container.size() >= _capacity) {
            _container.pop_front();
        }

        _container.push_back(val);
    }

    bool Empty() { return _container.empty(); }

    void Clear() { _container.clear(); }

    void SetCapacity(uint32_t capacity) { _capacity = capacity; }

   private:
    std::deque<T> _container;
    uint32_t _capacity;
};

template <typename T>
class CVSizeLimitQueue {
   public:
    CVSizeLimitQueue(uint32_t capacity = 5) : _queue(capacity) {}

    virtual ~CVSizeLimitQueue() {}

    void Exit() {
        _need_stop = true;
        _cv.notify_all();
    }

    void Clear() {
        std::lock_guard<std::mutex> lk(_mtx);

        _queue.Clear();
    }

    void EnableWrite(bool enable) {
        std::lock_guard<std::mutex> lk(_mtx);

        _enable_write = enable;
    }

    std::shared_ptr<T> GetLatestOneBlocking(bool erase, const uint32_t timeout_ms = UINT32_MAX) {
        std::unique_lock<std::mutex> recv_lk(_mtx);

        bool status = _cv.wait_for(recv_lk, std::chrono::milliseconds(timeout_ms),
                                   [this]() { return _need_stop || (!_queue.Empty()); });
        if (!status) {
            return nullptr;
        }

        return _queue.ReadLatestOne(erase);
    }

    std::shared_ptr<T> GetLatestOne(bool erase) {
        std::unique_lock<std::mutex> recv_lk(_mtx);

        return _queue.ReadLatestOne(erase);
    }

    std::vector<std::shared_ptr<T>> GetLatestNdata(const size_t n, bool erase) {
        std::unique_lock<std::mutex> recv_lk(_mtx);

        return _queue.ReadLatestN(n, erase);
    }

    virtual void PushOneAndNotify(T data) {
        std::unique_lock<std::mutex> recv_lk(_mtx);
        if (!_enable_write) {
            return;
        }

        _queue.PushOne(data);
        _cv.notify_all();
    }

    void SetCapacity(uint32_t capacity) { _queue.SetCapacity(capacity); }

   protected:
    std::mutex _mtx;
    std::condition_variable _cv;
    SizeLimitQueue<T> _queue;
    bool _enable_write = true;
    bool _need_stop = false;
};

}  // namespace adf
}  // namespace netaos
}  // namespace hozon