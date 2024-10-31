#ifndef CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_BUFFER_H_
#define CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_BUFFER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>

#include "framework/tools/framework_recorder/player/play_task.h"

namespace netaos {
namespace framework {
namespace record {

class PlayTaskBuffer {
 public:
  using TaskPtr = std::shared_ptr<PlayTask>;
  // if all tasks are in order, we can use other container to replace this
  using TaskMap = std::multimap<uint64_t, TaskPtr>;

  PlayTaskBuffer();
  virtual ~PlayTaskBuffer();

  size_t Size() const;
  bool Empty() const;

  void Push(const TaskPtr& task);
  TaskPtr Front();
  void PopFront();

 private:
  TaskMap tasks_;
  mutable std::mutex mutex_;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_PLAYER_PLAY_TASK_BUFFER_H_
