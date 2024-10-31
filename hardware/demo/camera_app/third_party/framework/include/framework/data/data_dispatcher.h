#ifndef CYBER_DATA_DATA_DISPATCHER_H_
#define CYBER_DATA_DATA_DISPATCHER_H_

#include <memory>
#include <mutex>
#include <vector>

#include "framework/common/log.h"
#include "framework/common/macros.h"
#include "framework/data/channel_buffer.h"
#include "framework/state.h"
#include "framework/time/time.h"

namespace netaos {
namespace framework {
namespace data {

using netaos::framework::Time;
using netaos::framework::base::AtomicHashMap;

template <typename T>
class DataDispatcher {
 public:
  using BufferVector =
      std::vector<std::weak_ptr<CacheBuffer<std::shared_ptr<T>>>>;
  ~DataDispatcher() {}

  void AddBuffer(const ChannelBuffer<T>& channel_buffer);

  bool Dispatch(const uint64_t channel_id, const std::shared_ptr<T>& msg);

 private:
  DataNotifier* notifier_ = DataNotifier::Instance();
  std::mutex buffers_map_mutex_;
  AtomicHashMap<uint64_t, BufferVector> buffers_map_;

  DECLARE_SINGLETON(DataDispatcher)
};

template <typename T>
inline DataDispatcher<T>::DataDispatcher() {}

template <typename T>
void DataDispatcher<T>::AddBuffer(const ChannelBuffer<T>& channel_buffer) {
  std::lock_guard<std::mutex> lock(buffers_map_mutex_);
  auto buffer = channel_buffer.Buffer();
  BufferVector* buffers = nullptr;
  if (buffers_map_.Get(channel_buffer.channel_id(), &buffers)) {
    buffers->emplace_back(buffer);
  } else {
    BufferVector new_buffers = {buffer};
    buffers_map_.Set(channel_buffer.channel_id(), new_buffers);
  }
}

template <typename T>
bool DataDispatcher<T>::Dispatch(const uint64_t channel_id,
                                 const std::shared_ptr<T>& msg) {
  BufferVector* buffers = nullptr;
  if (netaos::framework::IsShutdown()) {
    return false;
  }
  if (buffers_map_.Get(channel_id, &buffers)) {
    for (auto& buffer_wptr : *buffers) {
      if (auto buffer = buffer_wptr.lock()) {
        std::lock_guard<std::mutex> lock(buffer->Mutex());
        buffer->Fill(msg);
      }
    }
  } else {
    return false;
  }
  return notifier_->Notify(channel_id);
}

}  // namespace data
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_DATA_DATA_DISPATCHER_H_
