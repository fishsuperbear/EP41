#ifndef PYTHON_WRAPPER_PY_RECORD_H_
#define PYTHON_WRAPPER_PY_RECORD_H_

#include <unistd.h>

#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include "framework/message/protobuf_factory.h"
#include "framework/message/py_message.h"
#include "framework/message/raw_message.h"
#include "framework/proto/record.pb.h"
#include "framework/record/record_message.h"
#include "framework/record/record_reader.h"
#include "framework/record/record_writer.h"

namespace netaos {
namespace framework {
namespace record {

struct BagMessage {
  uint64_t timestamp = 0;
  std::string channel_name = "";
  std::string data = "";
  std::string data_type = "";
  bool end = true;
};

class PyRecordReader {
 public:
  explicit PyRecordReader(const std::string& file) {
    record_reader_.reset(new RecordReader(file));
  }

  BagMessage ReadMessage(
      uint64_t begin_time = 0,
      uint64_t end_time = std::numeric_limits<uint64_t>::max()) {
    BagMessage ret_msg;
    RecordMessage record_message;
    if (!record_reader_->ReadMessage(&record_message, begin_time, end_time)) {
      ret_msg.end = true;
      return ret_msg;
    }

    ret_msg.end = false;
    ret_msg.channel_name = record_message.channel_name;
    ret_msg.data = record_message.content;
    ret_msg.timestamp = record_message.time;
    ret_msg.data_type =
        record_reader_->GetMessageType(record_message.channel_name);
    return ret_msg;
  }

  uint64_t GetMessageNumber(const std::string& channel_name) {
    return record_reader_->GetMessageNumber(channel_name);
  }

  std::string GetMessageType(const std::string& channel_name) {
    return record_reader_->GetMessageType(channel_name);
  }

  std::string GetProtoDesc(const std::string& channel_name) {
    return record_reader_->GetProtoDesc(channel_name);
  }

  std::string GetHeaderString() {
    std::string org_data;
    record_reader_->GetHeader().SerializeToString(&org_data);
    return org_data;
  }

  void Reset() { record_reader_->Reset(); }

  std::set<std::string> GetChannelList() const {
    return record_reader_->GetChannelList();
  }

 private:
  std::unique_ptr<RecordReader> record_reader_;
};

class PyRecordWriter {
 public:
  bool Open(const std::string& path) { return record_writer_.Open(path); }

  void Close() { record_writer_.Close(); }

  bool WriteChannel(const std::string& channel_str, const std::string& type,
                    const std::string& proto_desc) {
    return record_writer_.WriteChannel(channel_str, type, proto_desc);
  }

  bool WriteMessage(const std::string& channel_name,
                    const std::string& rawmessage, uint64_t time,
                    const std::string& proto_desc = "") {
    return record_writer_.WriteMessage(
        channel_name, std::make_shared<message::RawMessage>(rawmessage), time,
        proto_desc);
  }

  bool SetSizeOfFileSegmentation(uint64_t size_kilobytes) {
    return record_writer_.SetSizeOfFileSegmentation(size_kilobytes);
  }

  bool SetIntervalOfFileSegmentation(uint64_t time_sec) {
    return record_writer_.SetIntervalOfFileSegmentation(time_sec);
  }

  uint64_t GetMessageNumber(const std::string& channel_name) const {
    return record_writer_.GetMessageNumber(channel_name);
  }

  const std::string& GetMessageType(const std::string& channel_name) const {
    return record_writer_.GetMessageType(channel_name);
  }

  const std::string& GetProtoDesc(const std::string& channel_name) const {
    return record_writer_.GetProtoDesc(channel_name);
  }

 private:
  RecordWriter record_writer_;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // PYTHON_WRAPPER_PY_RECORD_H_
