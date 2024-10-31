#ifndef CYBER_TRANSPORT_DISPATCHER_SHM_DISPATCHER_H_
#define CYBER_TRANSPORT_DISPATCHER_SHM_DISPATCHER_H_

#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>

#include "framework/base/atomic_rw_lock.h"
#include "framework/common/global_data.h"
#include "framework/common/log.h"
#include "framework/common/macros.h"
#include "framework/message/message_traits.h"
#include "framework/transport/dispatcher/dispatcher.h"
#include "framework/transport/shm/notifier_factory.h"
#include "framework/transport/shm/segment_factory.h"

namespace netaos {
namespace framework {
namespace transport {

class ShmDispatcher;
using ShmDispatcherPtr = ShmDispatcher*;
using netaos::framework::base::AtomicRWLock;
using netaos::framework::base::ReadLockGuard;
using netaos::framework::base::WriteLockGuard;

class ShmDispatcher : public Dispatcher {
 public:
  // key: channel_id
  using SegmentContainer = std::unordered_map<uint64_t, SegmentPtr>;

  virtual ~ShmDispatcher();

  void Shutdown() override;

  template <typename MessageT>
  void AddListener(const RoleAttributes& self_attr,
                   const MessageListener<MessageT>& listener);

  template <typename MessageT>
  void AddListener(const RoleAttributes& self_attr,
                   const RoleAttributes& opposite_attr,
                   const MessageListener<MessageT>& listener);

 private:
  void AddSegment(const RoleAttributes& self_attr);
  void ReadMessage(uint64_t channel_id, uint32_t block_index);
  void OnMessage(uint64_t channel_id, const std::shared_ptr<ReadableBlock>& rb,
                 const MessageInfo& msg_info);
  void ThreadFunc();
  bool Init();

  uint64_t host_id_;
  SegmentContainer segments_;
  std::unordered_map<uint64_t, uint32_t> previous_indexes_;
  AtomicRWLock segments_lock_;
  std::thread thread_;
  NotifierPtr notifier_;

  DECLARE_SINGLETON(ShmDispatcher)
};

template <typename MessageT>
void ShmDispatcher::AddListener(const RoleAttributes& self_attr,
                                const MessageListener<MessageT>& listener) {
  // FIXME: make it more clean
  auto listener_adapter = [listener](const std::shared_ptr<ReadableBlock>& rb,
                                     const MessageInfo& msg_info) {
    auto msg = std::make_shared<MessageT>();
    RETURN_IF(!message::ParseFromArray(
        rb->buf, static_cast<int>(rb->block->msg_size()), msg.get()));
    listener(msg, msg_info);
  };

  Dispatcher::AddListener<ReadableBlock>(self_attr, listener_adapter);
  AddSegment(self_attr);
}

template <typename MessageT>
void ShmDispatcher::AddListener(const RoleAttributes& self_attr,
                                const RoleAttributes& opposite_attr,
                                const MessageListener<MessageT>& listener) {
  // FIXME: make it more clean
  auto listener_adapter = [listener](const std::shared_ptr<ReadableBlock>& rb,
                                     const MessageInfo& msg_info) {
    auto msg = std::make_shared<MessageT>();
    RETURN_IF(!message::ParseFromArray(
        rb->buf, static_cast<int>(rb->block->msg_size()), msg.get()));
    listener(msg, msg_info);
  };

  Dispatcher::AddListener<ReadableBlock>(self_attr, opposite_attr,
                                         listener_adapter);
  AddSegment(self_attr);
}

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_DISPATCHER_SHM_DISPATCHER_H_
