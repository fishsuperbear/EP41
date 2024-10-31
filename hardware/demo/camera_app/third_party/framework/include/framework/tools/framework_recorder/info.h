#ifndef CYBER_TOOLS_CYBER_RECORDER_INFO_H_
#define CYBER_TOOLS_CYBER_RECORDER_INFO_H_

#include <chrono>
#include <iomanip>
#include <string>

#include "framework/common/time_conversion.h"
#include "framework/proto/record.pb.h"
#include "framework/record/file/record_file_reader.h"

using ::netaos::framework::common::UnixSecondsToString;

namespace netaos {
namespace framework {
namespace record {

class Info {
 public:
  Info();
  ~Info();
  bool Display(const std::string& file);
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_INFO_H_
