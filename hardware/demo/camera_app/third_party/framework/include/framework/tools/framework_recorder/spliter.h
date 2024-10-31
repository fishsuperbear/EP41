#ifndef CYBER_TOOLS_CYBER_RECORDER_SPLITER_H_
#define CYBER_TOOLS_CYBER_RECORDER_SPLITER_H_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/log.h"
#include "framework/proto/record.pb.h"
#include "framework/record/file/record_file_reader.h"
#include "framework/record/file/record_file_writer.h"
#include "framework/record/header_builder.h"

using ::netaos::framework::proto::ChannelCache;
using ::netaos::framework::proto::ChunkBody;
using ::netaos::framework::proto::ChunkHeader;
using ::netaos::framework::proto::Header;

namespace netaos {
namespace framework {
namespace record {

class Spliter {
 public:
  Spliter(const std::string& input_file, const std::string& output_file,
          const std::vector<std::string>& white_channels,
          const std::vector<std::string>& black_channels,
          uint64_t begin_time = 0,
          uint64_t end_time = std::numeric_limits<uint64_t>::max());
  virtual ~Spliter();
  bool Proc();

 private:
  RecordFileReader reader_;
  RecordFileWriter writer_;
  std::string input_file_;
  std::string output_file_;
  std::vector<std::string> white_channels_;
  std::vector<std::string> black_channels_;
  bool all_channels_;
  uint64_t begin_time_;
  uint64_t end_time_;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_SPLITER_H_
