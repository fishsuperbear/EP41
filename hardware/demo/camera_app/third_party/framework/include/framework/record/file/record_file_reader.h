#ifndef CYBER_RECORD_FILE_RECORD_FILE_READER_H_
#define CYBER_RECORD_FILE_RECORD_FILE_READER_H_

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include <limits>
#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "framework/common/log.h"
#include "framework/record/file/record_file_base.h"
#include "framework/record/file/section.h"
#include "framework/time/time.h"

namespace netaos {
namespace framework {
namespace record {

using google::protobuf::io::CodedInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::ZeroCopyInputStream;

class RecordFileReader : public RecordFileBase {
 public:
  RecordFileReader() = default;
  virtual ~RecordFileReader();
  bool Open(const std::string& path) override;
  void Close() override;
  bool Reset();
  bool ReadSection(Section* section);
  bool SkipSection(int64_t size);
  template <typename T>
  bool ReadSection(int64_t size, T* message);
  bool ReadIndex();
  bool EndOfFile() { return end_of_file_; }

 private:
  bool ReadHeader();
  bool end_of_file_ = false;
};

template <typename T>
bool RecordFileReader::ReadSection(int64_t size, T* message) {
  if (size < std::numeric_limits<int>::min() ||
      size > std::numeric_limits<int>::max()) {
    AERROR << "Size value greater than the range of int value.";
    return false;
  }
  FileInputStream raw_input(fd_, static_cast<int>(size));
  CodedInputStream coded_input(&raw_input);
  CodedInputStream::Limit limit = coded_input.PushLimit(static_cast<int>(size));
  if (!message->ParseFromCodedStream(&coded_input)) {
    AERROR << "Parse section message failed.";
    end_of_file_ = coded_input.ExpectAtEnd();
    return false;
  }
  if (!coded_input.ConsumedEntireMessage()) {
    AERROR << "Do not consumed entire message.";
    return false;
  }
  coded_input.PopLimit(limit);
  if (static_cast<int64_t>(message->ByteSizeLong()) != size) {
    AERROR << "Message size is not consistent in section header"
           << ", expect: " << size << ", actual: " << message->ByteSizeLong();
    return false;
  }
  return true;
}

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_RECORD_FILE_RECORD_FILE_READER_H_
