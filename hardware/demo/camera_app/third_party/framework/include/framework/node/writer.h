#ifndef CYBER_NODE_WRITER_H_
#define CYBER_NODE_WRITER_H_

#include <memory>
#include <string>
#include <vector>

#include "framework/proto/topology_change.pb.h"

#include "framework/common/log.h"
#include "framework/node/writer_base.h"
#include "framework/service_discovery/topology_manager.h"
#include "framework/transport/transport.h"

#ifdef ENABLE_TRACE
#include "framework/trace/trace_util.h"
#endif

namespace netaos {
namespace framework {

/**
 * @class Writer<MessageT>
 * @brief The Channel Writer has only one function: publish message through the
 * channel pointed in its RoleAttributes
 *
 * @tparam MessageT Message Type of the Writer handles
 */
template <typename MessageT>
class Writer : public WriterBase {
 public:
  using TransmitterPtr = std::shared_ptr<transport::Transmitter<MessageT>>;
  using ChangeConnection =
      typename service_discovery::Manager::ChangeConnection;

  /**
   * @brief Construct a new Writer object
   *
   * @param role_attr we use RoleAttributes to identify a Writer
   */
  explicit Writer(const proto::RoleAttributes& role_attr);
  virtual ~Writer();

  /**
   * @brief Init the Writer
   *
   * @return true if init successfully
   * @return false if init failed
   */
  bool Init() override;

  /**
   * @brief Shutdown the Writer
   */
  void Shutdown() override;

  /**
   * @brief Write a MessageT instance
   *
   * @param msg the message we want to write
   * @return true if write successfully
   * @return false if write failed
   */
  virtual bool Write(const MessageT& msg);

  /**
   * @brief Write a shared ptr of MessageT
   *
   * @param msg_ptr the message shared ptr we want to write
   * @return true if write successfully
   * @return false if write failed
   */
  virtual bool Write(const std::shared_ptr<MessageT>& msg_ptr);

  /**
   * @brief Is there any Reader that subscribes our Channel?
   * You can publish message when this return true
   *
   * @return true if the channel has reader
   * @return false if the channel has no reader
   */
  bool HasReader() override;

  /**
   * @brief Get all Readers that subscriber our writing channel
   *
   * @param readers vector result of RoleAttributes
   */
  void GetReaders(std::vector<proto::RoleAttributes>* readers) override;

#ifdef ENABLE_TRACE
  TraceContext& get_trace_context() {
    return m_context;
  }
#endif

 private:
  void JoinTheTopology();
  void LeaveTheTopology();
  void OnChannelChange(const proto::ChangeMsg& change_msg);

  TransmitterPtr transmitter_;

  ChangeConnection change_conn_;
  service_discovery::ChannelManagerPtr channel_manager_;

#ifdef ENABLE_TRACE
  TraceContext m_context;
#endif
};

template<typename MessageT>
Writer<MessageT>::Writer(const proto::RoleAttributes &role_attr)
    : WriterBase(role_attr), transmitter_(nullptr), channel_manager_(nullptr) {}

template <typename MessageT>
Writer<MessageT>::~Writer() {
  Shutdown();
}

template <typename MessageT>
bool Writer<MessageT>::Init() {
  {
    std::lock_guard<std::mutex> g(lock_);
    if (init_) {
      return true;
    }
    transmitter_ =
        transport::Transport::Instance()->CreateTransmitter<MessageT>(
            role_attr_);
    if (transmitter_ == nullptr) {
      return false;
    }
    init_ = true;
  }
  this->role_attr_.set_id(transmitter_->id().HashValue());
  channel_manager_ =
      service_discovery::TopologyManager::Instance()->channel_manager();
  JoinTheTopology();
  return true;
}

template <typename MessageT>
void Writer<MessageT>::Shutdown() {
  {
    std::lock_guard<std::mutex> g(lock_);
    if (!init_) {
      return;
    }
    init_ = false;
  }
  LeaveTheTopology();
  transmitter_ = nullptr;
  channel_manager_ = nullptr;
}

template <typename MessageT>
bool Writer<MessageT>::Write(const MessageT& msg) {
  RETURN_VAL_IF(!WriterBase::IsInit(), false);
  auto msg_ptr = std::make_shared<MessageT>(msg);
  return Write(msg_ptr);
}

template <typename MessageT>
bool Writer<MessageT>::Write(const std::shared_ptr<MessageT>& msg_ptr) {
  RETURN_VAL_IF(!WriterBase::IsInit(), false);
#ifdef ENABLE_TRACE
  TRACE_ENTRY_EXT(FrameworkTraceModule::WRITERWRITE, m_context);
  TRACE_SEND_MSG(msg_ptr);
  TRACE_SYS_PERF();
#endif
  return transmitter_->Transmit(msg_ptr);
}

template <typename MessageT>
void Writer<MessageT>::JoinTheTopology() {
  // add listener
  change_conn_ = channel_manager_->AddChangeListener(std::bind(
      &Writer<MessageT>::OnChannelChange, this, std::placeholders::_1));

  // get peer readers
  const std::string& channel_name = this->role_attr_.channel_name();
  std::vector<proto::RoleAttributes> readers;
  channel_manager_->GetReadersOfChannel(channel_name, &readers);
  for (auto& reader : readers) {
    transmitter_->Enable(reader);
  }

  channel_manager_->Join(this->role_attr_, proto::RoleType::ROLE_WRITER,
                         message::HasSerializer<MessageT>::value);
}

template <typename MessageT>
void Writer<MessageT>::LeaveTheTopology() {
  channel_manager_->RemoveChangeListener(change_conn_);
  channel_manager_->Leave(this->role_attr_, proto::RoleType::ROLE_WRITER);
}

template <typename MessageT>
void Writer<MessageT>::OnChannelChange(const proto::ChangeMsg& change_msg) {
  if (change_msg.role_type() != proto::RoleType::ROLE_READER) {
    return;
  }

  auto& reader_attr = change_msg.role_attr();
  if (reader_attr.channel_name() != this->role_attr_.channel_name()) {
    return;
  }

  auto operate_type = change_msg.operate_type();
  if (operate_type == proto::OperateType::OPT_JOIN) {
    transmitter_->Enable(reader_attr);
  } else {
    transmitter_->Disable(reader_attr);
  }
}

template <typename MessageT>
bool Writer<MessageT>::HasReader() {
  RETURN_VAL_IF(!WriterBase::IsInit(), false);
  return channel_manager_->HasReader(role_attr_.channel_name());
}

template <typename MessageT>
void Writer<MessageT>::GetReaders(std::vector<proto::RoleAttributes>* readers) {
  if (readers == nullptr) {
    return;
  }

  if (!WriterBase::IsInit()) {
    return;
  }

  channel_manager_->GetReadersOfChannel(role_attr_.channel_name(), readers);
}

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_NODE_WRITER_H_
