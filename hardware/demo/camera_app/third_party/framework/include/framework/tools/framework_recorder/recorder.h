#ifndef CYBER_TOOLS_CYBER_RECORDER_RECORDER_H_
#define CYBER_TOOLS_CYBER_RECORDER_RECORDER_H_

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/base/signal.h"
#include "framework/cyber.h"
#include "framework/message/raw_message.h"
#include "framework/proto/record.pb.h"
#include "framework/proto/topology_change.pb.h"
#include "framework/record/record_writer.h"

using netaos::framework::Node;
using netaos::framework::ReaderBase;
using netaos::framework::base::Connection;
using netaos::framework::message::RawMessage;
using netaos::framework::proto::ChangeMsg;
using netaos::framework::proto::RoleAttributes;
using netaos::framework::proto::RoleType;
using netaos::framework::service_discovery::ChannelManager;
using netaos::framework::service_discovery::TopologyManager;

namespace netaos {
namespace framework {
namespace record {

class Recorder : public std::enable_shared_from_this<Recorder> {
 public:
  Recorder(const std::string& output, bool all_channels,
           const std::vector<std::string>& white_channels,
           const std::vector<std::string>& black_channels);
  Recorder(const std::string& output, bool all_channels,
           const std::vector<std::string>& white_channels,
           const std::vector<std::string>& black_channels,
           const proto::Header& header);
  ~Recorder();
  bool Start();
  bool Stop();

 private:
  bool is_started_ = false;
  bool is_stopping_ = false;
  std::shared_ptr<Node> node_ = nullptr;
  std::shared_ptr<RecordWriter> writer_ = nullptr;
  std::shared_ptr<std::thread> display_thread_ = nullptr;
  Connection<const ChangeMsg&> change_conn_;
  std::string output_;
  bool all_channels_ = true;
  std::vector<std::string> white_channels_;
  std::vector<std::string> black_channels_;
  proto::Header header_;
  std::unordered_map<std::string, std::shared_ptr<ReaderBase>>
      channel_reader_map_;
  uint64_t message_count_;
  uint64_t message_time_;

  bool InitReadersImpl();

  bool FreeReadersImpl();

  bool InitReaderImpl(const std::string& channel_name,
                      const std::string& message_type);

  void TopologyCallback(const ChangeMsg& msg);

  void ReaderCallback(const std::shared_ptr<RawMessage>& message,
                      const std::string& channel_name);

  void FindNewChannel(const RoleAttributes& role_attr);

  void ShowProgress();
};

}  // namespace record
}  // namespace framework
}  // namespace netaos

#endif  // CYBER_TOOLS_CYBER_RECORDER_RECORDER_H_
