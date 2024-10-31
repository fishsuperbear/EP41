#ifndef CYBER_RECORD_FILE_RECORD_FILE_BASE_H_
#define CYBER_RECORD_FILE_RECORD_FILE_BASE_H_

#include <mutex>
#include <string>

#include "framework/proto/record.pb.h"

namespace netaos {
namespace framework {
namespace record {

const int HEADER_LENGTH = 2048;

class RecordFileBase {
 public:
  RecordFileBase() = default;
  virtual ~RecordFileBase() = default;
  virtual bool Open(const std::string& path) = 0;
  virtual void Close() = 0;
  const std::string& GetPath() const { return path_; }
  const proto::Header& GetHeader() const { return header_; }
  const proto::Index& GetIndex() const { return index_; }
  int64_t CurrentPosition();
  bool SetPosition(int64_t position);

 protected:
  std::mutex mutex_;
  std::string path_;
  proto::Header header_;
  proto::Index index_;
  int fd_ = -1;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_RECORD_FILE_RECORD_FILE_BASE_H_
