#ifndef CYBER_TRANSPORT_COMMON_IDENTITY_H_
#define CYBER_TRANSPORT_COMMON_IDENTITY_H_

#include <cstdint>
#include <cstring>
#include <string>

namespace netaos {
namespace framework {
namespace transport {

constexpr uint8_t ID_SIZE = 8;

class Identity {
 public:
  explicit Identity(bool need_generate = true);
  Identity(const Identity& another);
  virtual ~Identity();

  Identity& operator=(const Identity& another);
  bool operator==(const Identity& another) const;
  bool operator!=(const Identity& another) const;

  std::string ToString() const;
  size_t Length() const;
  uint64_t HashValue() const;

  const char* data() const { return data_; }
  void set_data(const char* data) {
    if (data == nullptr) {
      return;
    }
    std::memcpy(data_, data, sizeof(data_));
    Update();
  }

 private:
  void Update();

  char data_[ID_SIZE];
  uint64_t hash_value_;
};

}  // namespace transport
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TRANSPORT_COMMON_IDENTITY_H_
