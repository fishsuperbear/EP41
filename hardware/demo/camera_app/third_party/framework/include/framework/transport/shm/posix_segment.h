#ifndef CYBER_TRANSPORT_SHM_POSIX_SEGMENT_H_
#define CYBER_TRANSPORT_SHM_POSIX_SEGMENT_H_

#include <string>

#include "framework/transport/shm/segment.h"

namespace netaos {
namespace framework {
namespace transport {

class PosixSegment : public Segment {
 public:
  explicit PosixSegment(uint64_t channel_id);
  virtual ~PosixSegment();

  static const char* Type() { return "posix"; }

 private:
  void Reset() override;
  bool Remove() override;
  bool OpenOnly() override;
  bool OpenOrCreate() override;

  std::string shm_name_;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_SHM_POSIX_SEGMENT_H_
