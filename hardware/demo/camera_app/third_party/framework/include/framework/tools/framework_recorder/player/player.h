#ifndef CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAYER_H_
#define CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAYER_H_

#include <atomic>
#include <memory>
#include <thread>

#include "framework/tools/framework_recorder/player/play_param.h"
#include "framework/tools/framework_recorder/player/play_task_buffer.h"
#include "framework/tools/framework_recorder/player/play_task_consumer.h"
#include "framework/tools/framework_recorder/player/play_task_producer.h"

namespace netaos {
namespace framework {
namespace record {

class Player {
 public:
  using ConsumerPtr = std::unique_ptr<PlayTaskConsumer>;
  using ProducerPtr = std::unique_ptr<PlayTaskProducer>;
  using TaskBufferPtr = std::shared_ptr<PlayTaskBuffer>;

  explicit Player(const PlayParam& play_param);
  virtual ~Player();

  bool Init();
  bool Start();
  bool Stop();

 private:
  void ThreadFunc_Term();

 private:
  std::atomic<bool> is_initialized_ = {false};
  std::atomic<bool> is_stopped_ = {false};
  std::atomic<bool> is_paused_ = {false};
  std::atomic<bool> is_playonce_ = {false};
  ConsumerPtr consumer_;
  ProducerPtr producer_;
  TaskBufferPtr task_buffer_;
  std::shared_ptr<std::thread> term_thread_ = nullptr;
  static const uint64_t kSleepIntervalMiliSec;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAYER_H_
