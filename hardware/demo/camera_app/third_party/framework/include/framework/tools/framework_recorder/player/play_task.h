#ifndef CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_H_
#define CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_H_

#include <atomic>
#include <cstdint>
#include <memory>

#include "framework/message/raw_message.h"
#include "framework/node/writer.h"

namespace netaos {
namespace framework {
namespace record {

class PlayTask {
 public:
  using MessagePtr = std::shared_ptr<message::RawMessage>;
  using WriterPtr = std::shared_ptr<Writer<message::RawMessage>>;

  PlayTask(const MessagePtr& msg, const WriterPtr& writer,
           uint64_t msg_real_time_ns, uint64_t msg_play_time_ns);
  virtual ~PlayTask() {}

  void Play();

  uint64_t msg_real_time_ns() const { return msg_real_time_ns_; }
  uint64_t msg_play_time_ns() const { return msg_play_time_ns_; }
  static uint64_t played_msg_num() { return played_msg_num_.load(); }

 private:
  MessagePtr msg_;
  WriterPtr writer_;
  uint64_t msg_real_time_ns_;
  uint64_t msg_play_time_ns_;

  static std::atomic<uint64_t> played_msg_num_;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_H_
