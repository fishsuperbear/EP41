#ifndef CYBER_TRANSPORT_SHM_CONDITION_NOTIFIER_H_
#define CYBER_TRANSPORT_SHM_CONDITION_NOTIFIER_H_

#include <sys/types.h>
#include <atomic>
#include <cstdint>

#include "framework/common/macros.h"
#include "framework/transport/shm/notifier_base.h"

namespace netaos {
namespace framework {
namespace transport {

const uint32_t kBufLength = 4096;

class ConditionNotifier : public NotifierBase {
  struct Indicator {
    std::atomic<uint64_t> next_seq = {0};
    ReadableInfo infos[kBufLength];
    uint64_t seqs[kBufLength] = {0};
  };

 public:
  virtual ~ConditionNotifier();

  void Shutdown() override;
  bool Notify(const ReadableInfo& info) override;
  bool Listen(int timeout_ms, ReadableInfo* info) override;

  static const char* Type() { return "condition"; }

 private:
  bool Init();
  bool OpenOrCreate();
  bool OpenOnly();
  bool Remove();
  void Reset();

  key_t key_ = 0;
  void* managed_shm_ = nullptr;
  size_t shm_size_ = 0;
  Indicator* indicator_ = nullptr;
  uint64_t next_seq_ = 0;
  std::atomic<bool> is_shutdown_ = {false};

  DECLARE_SINGLETON(ConditionNotifier)
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_SHM_CONDITION_NOTIFIER_H_
