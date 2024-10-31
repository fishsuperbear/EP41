#ifndef CYBER_TOOLS_CYBER_RECORDER_RECOVERER_H_
#define CYBER_TOOLS_CYBER_RECORDER_RECOVERER_H_

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "framework/common/log.h"
#include "framework/proto/record.pb.h"
#include "framework/record/file/record_file_reader.h"
#include "framework/record/file/record_file_writer.h"

using ::netaos::framework::proto::ChannelCache;
using ::netaos::framework::proto::ChunkBody;
using ::netaos::framework::proto::ChunkHeader;
using ::netaos::framework::proto::Header;

namespace netaos {
namespace framework {
namespace record {

class Recoverer {
 public:
  Recoverer(const std::string& input_file, const std::string& output_file);
  virtual ~Recoverer();
  bool Proc();

 private:
  RecordFileReader reader_;
  RecordFileWriter writer_;
  std::string input_file_;
  std::string output_file_;
  std::vector<std::string> channel_vec_;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_RECOVERER_H_
