#ifndef CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_PRODUCER_H_
#define CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_PRODUCER_H_

#include <atomic>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "framework/message/raw_message.h"
#include "framework/node/node.h"
#include "framework/node/writer.h"
#include "framework/record/record_reader.h"
#include "framework/tools/framework_recorder/player/play_param.h"
#include "framework/tools/framework_recorder/player/play_task_buffer.h"

namespace netaos {
namespace framework {
namespace record {

class PlayTaskProducer {
 public:
  using NodePtr = std::shared_ptr<Node>;
  using ThreadPtr = std::unique_ptr<std::thread>;
  using TaskBufferPtr = std::shared_ptr<PlayTaskBuffer>;
  using RecordReaderPtr = std::shared_ptr<RecordReader>;
  using WriterPtr = std::shared_ptr<Writer<message::RawMessage>>;
  using WriterMap = std::unordered_map<std::string, WriterPtr>;
  using MessageTypeMap = std::unordered_map<std::string, std::string>;

  PlayTaskProducer(const TaskBufferPtr& task_buffer,
                   const PlayParam& play_param);
  virtual ~PlayTaskProducer();

  bool Init();
  void Start();
  void Stop();

  const PlayParam& play_param() const { return play_param_; }
  bool is_stopped() const { return is_stopped_.load(); }

 private:
  bool ReadRecordInfo();
  bool UpdatePlayParam();
  bool CreateWriters();
  void ThreadFunc();

  PlayParam play_param_;
  TaskBufferPtr task_buffer_;
  ThreadPtr produce_th_;

  std::atomic<bool> is_initialized_;
  std::atomic<bool> is_stopped_;

  NodePtr node_;
  WriterMap writers_;
  MessageTypeMap msg_types_;
  std::vector<RecordReaderPtr> record_readers_;

  uint64_t earliest_begin_time_;
  uint64_t latest_end_time_;
  uint64_t total_msg_num_;

  static const uint32_t kMinTaskBufferSize;
  static const uint32_t kPreloadTimeSec;
  static const uint64_t kSleepIntervalNanoSec;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_PRODUCER_H_
