#ifndef CYBER_NODE_WRITER_BASE_H_
#define CYBER_NODE_WRITER_BASE_H_

#include <atomic>
#include <mutex>
#include <string>
#include <vector>

#include "framework/proto/role_attributes.pb.h"

namespace netaos {
namespace framework {

/**
 * @class WriterBase
 * @brief Base class for a Writer. A Writer is an object to send messages
 * through a 'Channel'
 * @warning One Writer can only write one channel.
 * But different writers can write through the same channel
 */
class WriterBase {
 public:
  /**
   * @brief Construct a new Writer Base object
   *
   * @param role_attr role attributes for this Writer
   */
  explicit WriterBase(const proto::RoleAttributes& role_attr)
      : role_attr_(role_attr), init_(false) {}
  virtual ~WriterBase() {}

  /**
   * @brief Init the Writer
   *
   * @return true if init success
   * @return false if init failed
   */
  virtual bool Init() = 0;

  /**
   * @brief Shutdown the Writer
   */
  virtual void Shutdown() = 0;

  /**
   * @brief Is there any Reader that subscribes our Channel?
   * You can publish message when this return true
   *
   * @return true if the channel has reader
   * @return false if the channel has no reader
   */
  virtual bool HasReader() { return false; }

  /**
   * @brief Get all Readers that subscriber our writing channel
   *
   * @param readers result vector of RoleAttributes
   */
  virtual void GetReaders(std::vector<proto::RoleAttributes>* readers) {}

  /**
   * @brief Get Writer's Channel name
   *
   * @return const std::string& const reference to the channel name
   */
  const std::string& GetChannelName() const {
    return role_attr_.channel_name();
  }

  /**
   * @brief Is Writer initialized?
   *
   * @return true if the Writer is inited
   * @return false if the Write is not inited
   */
  bool IsInit() const {
    std::lock_guard<std::mutex> g(lock_);
    return init_;
  }

 protected:
  proto::RoleAttributes role_attr_;
  mutable std::mutex lock_;
  bool init_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_NODE_WRITER_BASE_H_
