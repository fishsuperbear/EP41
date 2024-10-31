#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>

namespace hozon {
namespace fsmcore {

/*****************************************************************************/
/* counterdown timer has following stage                                     */
/*****************************************************************************/
enum class CounterdownStage : std::uint32_t {
  INIT = 0,  // 倒计时器构造完成，未开始倒计时
  SETTED,    // 倒计时器正在倒计时中
  CANCELED,  // 倒计时器在倒计时中被取消
  TIMEUP     // 倒计时器时间到，可以进行析构啦
};

/*****************************************************************************/
/* counterdown timer                                                         */
/*****************************************************************************/
class CounterdownTimer {
 public:
  explicit CounterdownTimer(const std::string name, uint32_t time_out = 1000)
      : _name(name), _is_canceled(false), _timeout(time_out) {
    _state = CounterdownStage::INIT;
  }
  ~CounterdownTimer() {
    release_task();
    if (_task.joinable()) {
      _task.join();
    }
  }

  /***************************************************************************/
  /* if not timeup or timeup, can call this function to make _task terminate */
  /***************************************************************************/
  void release_task() {
    _cv.notify_all();  // 最多只有一个倒计时器，notify_one() 也可以
  }

  /***************************************************************************/
  /* trigger timer start to count                                            */
  /***************************************************************************/
  void settering() {
    if (_state == CounterdownStage::INIT) {
      _task = std::thread([&]() {
        std::unique_lock<std::mutex> lk(_mtx);
        auto flag =
            _cv.wait_for(lk, _timeout, [&]() { return _is_canceled.load(); });
        if (!flag) {
          _state = CounterdownStage::TIMEUP;
        } else {
          _state = CounterdownStage::CANCELED;
        }
      });
      _state = CounterdownStage::SETTED;
    }
  }

  void canceling() {
    if (_state == CounterdownStage::SETTED) {
      _is_canceled.store(true);
      release_task();
      if (_task.joinable()) {
        _task.join();
      }
      _state = CounterdownStage::CANCELED;
    }
  }

  CounterdownStage get_state() { return _state; }

 private:
  std::string _name;
  std::mutex _mtx;
  std::thread _task;  // 一个倒计时器，最多只有一个
  CounterdownStage _state;
  std::condition_variable _cv;
  std::atomic<bool> _is_canceled;
  std::chrono::milliseconds _timeout;
};

}  // namespace fsmcore
}  // namespace hozon