#ifndef CYBER_TRANSPORT_MESSAGE_MESSAGE_INFO_H_
#define CYBER_TRANSPORT_MESSAGE_MESSAGE_INFO_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "framework/transport/common/identity.h"

namespace netaos {
namespace framework {
namespace transport {

class MessageInfo {
 public:
  MessageInfo();
  MessageInfo(const Identity& sender_id, uint64_t seq_num);
  MessageInfo(const Identity& sender_id, uint64_t seq_num,
              const Identity& spare_id);
  MessageInfo(const MessageInfo& another);
  virtual ~MessageInfo();

  MessageInfo& operator=(const MessageInfo& another);
  bool operator==(const MessageInfo& another) const;
  bool operator!=(const MessageInfo& another) const;

  bool SerializeTo(std::string* dst) const;
  bool SerializeTo(char* dst, std::size_t len) const;
  bool DeserializeFrom(const std::string& src);
  bool DeserializeFrom(const char* src, std::size_t len);

  // getter and setter
  const Identity& sender_id() const { return sender_id_; }
  void set_sender_id(const Identity& sender_id) { sender_id_ = sender_id; }

  uint64_t channel_id() const { return channel_id_; }
  void set_channel_id(uint64_t channel_id) { channel_id_ = channel_id; }

  uint64_t seq_num() const { return seq_num_; }
  void set_seq_num(uint64_t seq_num) { seq_num_ = seq_num; }

  const Identity& spare_id() const { return spare_id_; }
  void set_spare_id(const Identity& spare_id) { spare_id_ = spare_id; }

  static const std::size_t kSize;

 private:
  Identity sender_id_;
  uint64_t channel_id_ = 0;
  uint64_t seq_num_ = 0;
  Identity spare_id_;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_MESSAGE_MESSAGE_INFO_H_
