#ifndef CYBER_TRANSPORT_SHM_SEGMENT_FACTORY_H_
#define CYBER_TRANSPORT_SHM_SEGMENT_FACTORY_H_

#include "framework/transport/shm/segment.h"

namespace netaos {
namespace framework {
namespace transport {

class SegmentFactory {
 public:
  static SegmentPtr CreateSegment(uint64_t channel_id);
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_SHM_SEGMENT_FACTORY_H_
