#ifndef CYBER_COMMON_TYPES_H_
#define CYBER_COMMON_TYPES_H_

#include <cstdint>

namespace netaos {
namespace framework {

class NullType {};

// Return code definition for framework internal function return.
enum ReturnCode {
  SUCC = 0,
  FAIL = 1,
};

/**
 * @brief Describe relation between nodes, writers/readers...
 */
enum Relation : std::uint8_t {
  NO_RELATION = 0,
  DIFF_HOST,  // different host
  DIFF_PROC,  // same host, but different process
  SAME_PROC,  // same process
};

static const char SRV_CHANNEL_REQ_SUFFIX[] = "__SRV__REQUEST";
static const char SRV_CHANNEL_RES_SUFFIX[] = "__SRV__RESPONSE";

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_COMMON_TYPES_H_
