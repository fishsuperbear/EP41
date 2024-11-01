#ifndef CYBER_RECORD_RECORD_READER_H_
#define CYBER_RECORD_RECORD_READER_H_

#include <limits>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "framework/proto/record.pb.h"

#include "framework/record/file/record_file_reader.h"
#include "framework/record/record_base.h"
#include "framework/record/record_message.h"

namespace netaos {
namespace framework {
namespace record {

/**
 * @brief The record reader.
 */
class RecordReader : public RecordBase {
 public:
  using FileReaderPtr = std::unique_ptr<RecordFileReader>;
  using ChannelInfoMap = std::unordered_map<std::string, proto::ChannelCache>;

  /**
   * @brief The constructor with record file path as parameter.
   *
   * @param file
   */
  explicit RecordReader(const std::string& file);

  /**
   * @brief The destructor.
   */
  virtual ~RecordReader();

  /**
   * @brief Is this record reader is valid.
   *
   * @return True for valid, false for not.
   */
  bool IsValid() const { return is_valid_; }

  /**
   * @brief Read one message from reader.
   *
   * @param message
   * @param begin_time
   * @param end_time
   *
   * @return True for success, false for not.
   */
  bool ReadMessage(RecordMessage* message, uint64_t begin_time = 0,
                   uint64_t end_time = std::numeric_limits<uint64_t>::max());

  /**
   * @brief Reset the message index of record reader.
   */
  void Reset();

  /**
   * @brief Get message number by channel name.
   *
   * @param channel_name
   *
   * @return Message number.
   */
  uint64_t GetMessageNumber(const std::string& channel_name) const override;

  /**
   * @brief Get message type by channel name.
   *
   * @param channel_name
   *
   * @return Message type.
   */
  const std::string& GetMessageType(
      const std::string& channel_name) const override;

  /**
   * @brief Get proto descriptor string by channel name.
   *
   * @param channel_name
   *
   * @return Proto descriptor string by channel name.
   */
  const std::string& GetProtoDesc(
      const std::string& channel_name) const override;

  /**
   * @brief Get channel list.
   *
   * @return List container with all channel name string.
   */
  std::set<std::string> GetChannelList() const override;

 private:
  bool ReadNextChunk(uint64_t begin_time, uint64_t end_time);

  bool is_valid_ = false;
  bool reach_end_ = false;
  std::unique_ptr<proto::ChunkBody> chunk_ = nullptr;
  proto::Index index_;
  int message_index_ = 0;
  ChannelInfoMap channel_info_;
  FileReaderPtr file_reader_;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_RECORD_RECORD_READER_H_
