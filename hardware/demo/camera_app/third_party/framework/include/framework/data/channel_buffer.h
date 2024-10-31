#ifndef CYBER_DATA_CHANNEL_BUFFER_H_
#define CYBER_DATA_CHANNEL_BUFFER_H_

#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

#include "framework/common/global_data.h"
#include "framework/common/log.h"
#include "framework/data/data_notifier.h"

namespace netaos {
namespace framework {
namespace data {

using netaos::framework::common::GlobalData;

template <typename T>
class ChannelBuffer {
 public:
  using BufferType = CacheBuffer<std::shared_ptr<T>>;
  ChannelBuffer(uint64_t channel_id, BufferType* buffer)
      : channel_id_(channel_id), buffer_(buffer) {}

  bool Fetch(uint64_t* index, std::shared_ptr<T>& m);  // NOLINT

  bool Latest(std::shared_ptr<T>& m);  // NOLINT

  bool FetchMulti(uint64_t fetch_size, std::vector<std::shared_ptr<T>>* vec);

  uint64_t channel_id() const { return channel_id_; }
  std::shared_ptr<BufferType> Buffer() const { return buffer_; }

 private:
  uint64_t channel_id_;
  std::shared_ptr<BufferType> buffer_;
};

template <typename T>
bool ChannelBuffer<T>::Fetch(uint64_t* index,
                             std::shared_ptr<T>& m) {  // NOLINT
  std::lock_guard<std::mutex> lock(buffer_->Mutex());
  if (buffer_->Empty()) {
    return false;
  }

  if (*index == 0) {
    *index = buffer_->Tail();
  } else if (*index == buffer_->Tail() + 1) {
    return false;
  } else if (*index < buffer_->Head()) {
    auto interval = buffer_->Tail() - *index;
    AWARN << "channel[" << GlobalData::GetChannelById(channel_id_) << "] "
          << "read buffer overflow, drop_message[" << interval << "] pre_index["
          << *index << "] current_index[" << buffer_->Tail() << "] ";
    *index = buffer_->Tail();
  }
  m = buffer_->at(*index);
  return true;
}

template <typename T>
bool ChannelBuffer<T>::Latest(std::shared_ptr<T>& m) {  // NOLINT
  std::lock_guard<std::mutex> lock(buffer_->Mutex());
  if (buffer_->Empty()) {
    return false;
  }

  m = buffer_->Back();
  return true;
}

template <typename T>
bool ChannelBuffer<T>::FetchMulti(uint64_t fetch_size,
                                  std::vector<std::shared_ptr<T>>* vec) {
  std::lock_guard<std::mutex> lock(buffer_->Mutex());
  if (buffer_->Empty()) {
    return false;
  }

  auto num = std::min(buffer_->Size(), fetch_size);
  vec->reserve(num);
  for (auto index = buffer_->Tail() - num + 1; index <= buffer_->Tail();
       ++index) {
    vec->emplace_back(buffer_->at(index));
  }
  return true;
}

}  // namespace data
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_DATA_CHANNEL_BUFFER_H_
