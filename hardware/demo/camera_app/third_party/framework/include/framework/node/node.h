#ifndef CYBER_NODE_NODE_H_
#define CYBER_NODE_NODE_H_

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "framework/node/node_channel_impl.h"
#include "framework/node/node_service_impl.h"

#ifdef ENABLE_TRACE
#include "framework/trace/trace_util.h"
#endif

namespace netaos {
namespace framework {

template <typename M0, typename M1, typename M2, typename M3>
class Component;
class TimerComponent;

/**
 * @class Node
 * @brief Node is the fundamental building block of netaos framework.
 * every module contains and communicates through the node.
 * A module can have different types of communication by defining
 * read/write and/or service/client in a node.
 * @warning Duplicate name is not allowed in topo objects, such as node,
 * reader/writer, service/clinet in the topo.
 */
class Node {
 public:
  template <typename M0, typename M1, typename M2, typename M3>
  friend class Component;
  friend class TimerComponent;
  friend bool Init(const char*);
  friend std::unique_ptr<Node> CreateNode(const std::string&,
                                          const std::string&);
  virtual ~Node();

  /**
   * @brief Get node's name.
   * @warning duplicate node name is not allowed in the topo.
   */
  const std::string& Name() const;

  /**
   * @brief Create a Writer with specific message type.
   *
   * @tparam MessageT Message Type
   * @param role_attr is a protobuf message RoleAttributes, which includes the
   * channel name and other info.
   * @return std::shared_ptr<Writer<MessageT>> result Writer Object
   */
  template <typename MessageT>
  auto CreateWriter(const proto::RoleAttributes& role_attr
#ifdef ENABLE_TRACE
                    ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                   ) -> std::shared_ptr<Writer<MessageT>>;

  /**
   * @brief Create a Writer with specific message type.
   *
   * @tparam MessageT Message Type
   * @param channel_name the channel name to be published.
   * @return std::shared_ptr<Writer<MessageT>> result Writer Object
   */
  template <typename MessageT>
  auto CreateWriter(const std::string& channel_name
#ifdef ENABLE_TRACE
                    ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                   ) -> std::shared_ptr<Writer<MessageT>>;

  /**
   * @brief Create a Reader with specific message type with channel name
   * qos and other configs used will be default
   *
   * @tparam MessageT Message Type
   * @param channel_name the channel of the reader subscribed.
   * @param reader_func invoked when message receive
   * invoked when the message is received.
   * @return std::shared_ptr<framework::Reader<MessageT>> result Reader Object
   */
  template <typename MessageT>
  auto CreateReader(const std::string& channel_name,
                    const CallbackFunc<MessageT>& reader_func = nullptr
#ifdef ENABLE_TRACE
                    ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                   ) -> std::shared_ptr<framework::Reader<MessageT>>;

  /**
   * @brief Create a Reader with specific message type with reader config
   *
   * @tparam MessageT Message Type
   * @param config instance of `ReaderConfig`,
   * include channel name, qos and pending queue size
   * @param reader_func invoked when message receive
   * @return std::shared_ptr<framework::Reader<MessageT>> result Reader Object
   */
  template <typename MessageT>
  auto CreateReader(const ReaderConfig& config,
                    const CallbackFunc<MessageT>& reader_func = nullptr
#ifdef ENABLE_TRACE
                    ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                   ) -> std::shared_ptr<framework::Reader<MessageT>>;

  /**
   * @brief Create a Reader object with `RoleAttributes`
   *
   * @tparam MessageT Message Type
   * @param role_attr instance of `RoleAttributes`,
   * includes channel name, qos, etc.
   * @param reader_func invoked when message receive
   * @return std::shared_ptr<framework::Reader<MessageT>> result Reader Object
   */
  template <typename MessageT>
  auto CreateReader(const proto::RoleAttributes& role_attr,
                    const CallbackFunc<MessageT>& reader_func = nullptr
#ifdef ENABLE_TRACE
                    ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                   ) -> std::shared_ptr<framework::Reader<MessageT>>;

  /**
   * @brief Create a Service object with specific `service_name`
   *
   * @tparam Request Message Type of the Request
   * @tparam Response Message Type of the Response
   * @param service_name specific service name to a serve
   * @param service_callback invoked when a service is called
   * @return std::shared_ptr<Service<Request, Response>> result `Service`
   */
  template <typename Request, typename Response>
  auto CreateService(const std::string& service_name,
                     const typename Service<Request, Response>::ServiceCallback& service_callback
#ifdef ENABLE_TRACE
                     ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                    ) -> std::shared_ptr<Service<Request, Response>>;

  /**
   * @brief Create a Client object to request Service with `service_name`
   *
   * @tparam Request Message Type of the Request
   * @tparam Response Message Type of the Response
   * @param service_name specific service name to a Service
   * @return std::shared_ptr<Client<Request, Response>> result `Client`
   */
  template <typename Request, typename Response>
  auto CreateClient(const std::string& service_name
#ifdef ENABLE_TRACE
                    ,const SourceInfo& src_info = SourceInfo::get_current_source_info()
#endif
                   ) -> std::shared_ptr<Client<Request, Response>>;

  bool DeleteReader(const std::string& channel_name);
  bool DeleteReader(const ReaderConfig& config);
  bool DeleteReader(const proto::RoleAttributes& role_attr);
  /**
   * @brief Observe all readers' data
   */
  void Observe();

  /**
   * @brief clear all readers' data
   */
  void ClearData();

  /**
   * @brief Get the Reader object that subscribe `channel_name`
   *
   * @tparam MessageT Message Type
   * @param channel_name channel name
   * @return std::shared_ptr<Reader<MessageT>> result reader
   */
  template <typename MessageT>
  auto GetReader(const std::string& channel_name)
      -> std::shared_ptr<Reader<MessageT>>;

 private:
  explicit Node(const std::string& node_name,
                const std::string& name_space = "");

  std::string node_name_;
  std::string name_space_;

  std::mutex readers_mutex_;
  std::map<std::string, std::shared_ptr<ReaderBase>> readers_;

  std::unique_ptr<NodeChannelImpl> node_channel_impl_ = nullptr;
  std::unique_ptr<NodeServiceImpl> node_service_impl_ = nullptr;

#ifdef ENABLE_TRACE
  TraceContext m_trace_context;
#endif
};

template <typename MessageT>
auto Node::CreateWriter(const proto::RoleAttributes& role_attr
#ifdef ENABLE_TRACE
                        ,const SourceInfo& src_info
#endif
                       ) -> std::shared_ptr<Writer<MessageT>> {
  auto writer = node_channel_impl_->template CreateWriter<MessageT>(role_attr);
#ifdef ENABLE_TRACE
  writer->get_trace_context().set_node_name(m_trace_context.get_node_name());
  writer->get_trace_context().set_attribute_info(role_attr.channel_name());
  writer->get_trace_context().set_role(TraceContext::Role::Writer);
  TRACE_ENTRY(FrameworkTraceModule::CREATEWRITER, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return writer;
}

template <typename MessageT>
auto Node::CreateWriter(const std::string& channel_name
#ifdef ENABLE_TRACE
                        ,const SourceInfo& src_info
#endif
                       ) -> std::shared_ptr<Writer<MessageT>> {
  auto writer = node_channel_impl_->template CreateWriter<MessageT>(channel_name);
#ifdef ENABLE_TRACE
  writer->get_trace_context().set_node_name(m_trace_context.get_node_name());
  writer->get_trace_context().set_attribute_info(channel_name);
  writer->get_trace_context().set_role(TraceContext::Role::Writer);
  TRACE_ENTRY(FrameworkTraceModule::CREATEWRITER, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return writer;
}

template <typename MessageT>
auto Node::CreateReader(const proto::RoleAttributes& role_attr,
                        const CallbackFunc<MessageT>& reader_func
#ifdef ENABLE_TRACE
                        ,const SourceInfo& src_info
#endif
                       ) -> std::shared_ptr<Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  if (readers_.find(role_attr.channel_name()) != readers_.end()) {
    AWARN << "Failed to create reader: reader with the same channel already "
             "exists.";
    return nullptr;
  }
  auto reader = node_channel_impl_->template CreateReader<MessageT>(
      role_attr, reader_func);
  if (reader != nullptr) {
    readers_.emplace(std::make_pair(role_attr.channel_name(), reader));
  }
#ifdef ENABLE_TRACE
  reader->get_trace_context().set_node_name(m_trace_context.get_node_name());
  reader->get_trace_context().set_attribute_info(role_attr.channel_name());
  reader->get_trace_context().set_role(TraceContext::Role::Reader);
  reader->get_trace_context().set_source_info(src_info);
  TRACE_ENTRY(FrameworkTraceModule::CREATEREADER, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return reader;
}

template <typename MessageT>
auto Node::CreateReader(const ReaderConfig& config,
                        const CallbackFunc<MessageT>& reader_func
#ifdef ENABLE_TRACE
                        ,const SourceInfo& src_info
#endif
                       ) -> std::shared_ptr<framework::Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  if (readers_.find(config.channel_name) != readers_.end()) {
    AWARN << "Failed to create reader: reader with the same channel already "
             "exists.";
    return nullptr;
  }
  auto reader =
      node_channel_impl_->template CreateReader<MessageT>(config, reader_func);
  if (reader != nullptr) {
    readers_.emplace(std::make_pair(config.channel_name, reader));
  }
#ifdef ENABLE_TRACE
  reader->get_trace_context().set_node_name(m_trace_context.get_node_name());
  reader->get_trace_context().set_attribute_info(config.channel_name);
  reader->get_trace_context().set_role(TraceContext::Role::Reader);
  reader->get_trace_context().set_source_info(src_info);
  TRACE_ENTRY(FrameworkTraceModule::CREATEREADER, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return reader;
}

template <typename MessageT>
auto Node::CreateReader(const std::string& channel_name,
                        const CallbackFunc<MessageT>& reader_func
#ifdef ENABLE_TRACE
                        ,const SourceInfo& src_info
#endif
                       ) -> std::shared_ptr<Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  if (readers_.find(channel_name) != readers_.end()) {
    AWARN << "Failed to create reader: reader with the same channel already "
             "exists.";
    return nullptr;
  }
  auto reader = node_channel_impl_->template CreateReader<MessageT>(
      channel_name, reader_func);
  if (reader != nullptr) {
    readers_.emplace(std::make_pair(channel_name, reader));
  }
#ifdef ENABLE_TRACE
  reader->get_trace_context().set_node_name(m_trace_context.get_node_name());
  reader->get_trace_context().set_attribute_info(channel_name);
  reader->get_trace_context().set_role(TraceContext::Role::Reader);
  reader->get_trace_context().set_source_info(src_info);
  TRACE_ENTRY(FrameworkTraceModule::CREATEREADER, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return reader;
}

template <typename Request, typename Response>
auto Node::CreateService(const std::string& service_name,
                         const typename Service<Request, Response>::ServiceCallback& service_callback
#ifdef ENABLE_TRACE
                         ,const SourceInfo& src_info
#endif
                        ) -> std::shared_ptr<Service<Request, Response>> {
  auto service = node_service_impl_->template CreateService<Request, Response>(
      service_name, service_callback);
#ifdef ENABLE_TRACE
  service->get_trace_context().set_node_name(m_trace_context.get_node_name());
  service->get_trace_context().set_attribute_info(service_name);
  service->get_trace_context().set_role(TraceContext::Role::Service);
  service->get_trace_context().set_source_info(src_info);
  TRACE_ENTRY(FrameworkTraceModule::CREATESERVICE, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return service;
}

template <typename Request, typename Response>
auto Node::CreateClient(const std::string& service_name
#ifdef ENABLE_TRACE
                        ,const SourceInfo& src_info
#endif
                       ) -> std::shared_ptr<Client<Request, Response>> {
  auto client = node_service_impl_->template CreateClient<Request, Response>(
      service_name);
#ifdef ENABLE_TRACE
  client->get_trace_context().set_node_name(m_trace_context.get_node_name());
  client->get_trace_context().set_attribute_info(service_name);
  client->get_trace_context().set_role(TraceContext::Role::Client);
  TRACE_ENTRY(FrameworkTraceModule::CREATECLIENT, src_info);
  TRACE_SYS_PERF();
  TRACE_SYS_INFO();
#endif
  return client;
}

template <typename MessageT>
auto Node::GetReader(const std::string& name)
    -> std::shared_ptr<Reader<MessageT>> {
  std::lock_guard<std::mutex> lg(readers_mutex_);
  auto it = readers_.find(name);
  if (it != readers_.end()) {
    return std::dynamic_pointer_cast<Reader<MessageT>>(it->second);
  }
  return nullptr;
}

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_NODE_NODE_H_
