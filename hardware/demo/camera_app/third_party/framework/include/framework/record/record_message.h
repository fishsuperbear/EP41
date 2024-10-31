#ifndef CYBER_RECORD_RECORD_MESSAGE_H_
#define CYBER_RECORD_RECORD_MESSAGE_H_

#include <cstdint>
#include <string>

namespace netaos {
namespace framework {
namespace record {

static constexpr size_t kGB = 1 << 30;
static constexpr size_t kMB = 1 << 20;
static constexpr size_t kKB = 1 << 10;

/**
 * @brief Basic data struct of record message.
 */
struct RecordMessage {
  /**
   * @brief The constructor.
   */
  RecordMessage() {}

  /**
   * @brief The constructor.
   *
   * @param name
   * @param message
   * @param msg_time
   */
  RecordMessage(const std::string& name, const std::string& message,
                uint64_t msg_time)
      : channel_name(name), content(message), time(msg_time) {}

  /**
   * @brief The channel name of the message.
   */
  std::string channel_name;

  /**
   * @brief The content of the message.
   */
  std::string content;

  /**
   * @brief The time (nanosecond) of the message.
   */
  uint64_t time;
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_RECORD_RECORD_READER_H_
