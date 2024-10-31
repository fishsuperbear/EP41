#ifndef CYBER_SERVICE_SERVICE_H_
#define CYBER_SERVICE_SERVICE_H_

#include <list>
#include <memory>
#include <string>
#include <utility>

#include "framework/common/types.h"
#include "framework/node/node_channel_impl.h"
#include "framework/scheduler/scheduler.h"
#include "framework/service/service_base.h"

#ifdef ENABLE_TRACE
#include "framework/trace/trace_util.h"
#endif

namespace netaos {
namespace framework {

/**
 * @class Service
 * @brief Service handles `Request` from the Client, and send a `Response` to
 * it.
 *
 * @tparam Request the request type
 * @tparam Response the response type
 */
template <typename Request, typename Response>
class Service : public ServiceBase {
 public:
  using ServiceCallback = std::function<void(const std::shared_ptr<Request>&,
                                             std::shared_ptr<Response>&)>;
  /**
   * @brief Construct a new Service object
   *
   * @param node_name used to fill RoleAttribute when join the topology
   * @param service_name the service name we provide
   * @param service_callback reference of `ServiceCallback` object
   */
  Service(const std::string& node_name, const std::string& service_name,
          const ServiceCallback& service_callback)
      : ServiceBase(service_name),
        node_name_(node_name),
        service_callback_(service_callback),
        request_channel_(service_name + SRV_CHANNEL_REQ_SUFFIX),
        response_channel_(service_name + SRV_CHANNEL_RES_SUFFIX) {}

  /**
   * @brief Construct a new Service object
   *
   * @param node_name used to fill RoleAttribute when join the topology
   * @param service_name the service name we provide
   * @param service_callback rvalue reference of `ServiceCallback` object
   */
  Service(const std::string& node_name, const std::string& service_name,
          ServiceCallback&& service_callback)
      : ServiceBase(service_name),
        node_name_(node_name),
        service_callback_(service_callback),
        request_channel_(service_name + SRV_CHANNEL_REQ_SUFFIX),
        response_channel_(service_name + SRV_CHANNEL_RES_SUFFIX) {}

  /**
   * @brief Forbid default constructing
   */
  Service() = delete;

  ~Service() { destroy(); }

  /**
   * @brief Init the Service
   */
  bool Init();

  /**
   * @brief Destroy the Service
   */
  void destroy();

#ifdef ENABLE_TRACE
  TraceContext& get_trace_context() {
    return m_context;
  }
#endif

 private:
  void HandleRequest(const std::shared_ptr<Request>& request,
                     const transport::MessageInfo& message_info);

  void SendResponse(const transport::MessageInfo& message_info,
                    const std::shared_ptr<Response>& response);

  bool IsInit(void) const { return request_receiver_ != nullptr; }

  std::string node_name_;
  ServiceCallback service_callback_;

  std::function<void(const std::shared_ptr<Request>&,
                     const transport::MessageInfo&)>
      request_callback_;
  std::shared_ptr<transport::Transmitter<Response>> response_transmitter_;
  std::shared_ptr<transport::Receiver<Request>> request_receiver_;
  std::string request_channel_;
  std::string response_channel_;
  std::mutex service_handle_request_mutex_;

  volatile bool inited_ = false;
  void Enqueue(std::function<void()>&& task);
  void Process();
  std::thread thread_;
  std::mutex queue_mutex_;
  std::condition_variable condition_;
  std::list<std::function<void()>> tasks_;

#ifdef ENABLE_TRACE
  TraceContext m_context;
#endif
};

template <typename Request, typename Response>
void Service<Request, Response>::destroy() {
  inited_ = false;
  {
    std::lock_guard<std::mutex> lg(queue_mutex_);
    this->tasks_.clear();
  }
  condition_.notify_all();
  if (thread_.joinable()) {
    thread_.join();
  }
}

template <typename Request, typename Response>
inline void Service<Request, Response>::Enqueue(std::function<void()>&& task) {
  std::lock_guard<std::mutex> lg(queue_mutex_);
  tasks_.emplace_back(std::move(task));
  condition_.notify_one();
}

template <typename Request, typename Response>
void Service<Request, Response>::Process() {
  while (!framework::IsShutdown()) {
    std::unique_lock<std::mutex> ul(queue_mutex_);
    condition_.wait(ul, [this]() { return !inited_ || !this->tasks_.empty(); });
    if (!inited_) {
      break;
    }
    if (!tasks_.empty()) {
      auto task = tasks_.front();
      tasks_.pop_front();
      ul.unlock();
      task();
    }
  }
}

template <typename Request, typename Response>
bool Service<Request, Response>::Init() {
  if (IsInit()) {
    return true;
  }
  proto::RoleAttributes role;
  role.set_node_name(node_name_);
  role.set_channel_name(response_channel_);
  auto channel_id = common::GlobalData::RegisterChannel(response_channel_);
  role.set_channel_id(channel_id);
  role.mutable_qos_profile()->CopyFrom(
      transport::QosProfileConf::QOS_PROFILE_SERVICES_DEFAULT);
  auto transport = transport::Transport::Instance();
  response_transmitter_ =
      transport->CreateTransmitter<Response>(role, proto::OptionalMode::RTPS);
  if (response_transmitter_ == nullptr) {
    AERROR << " Create response pub failed.";
    return false;
  }

  request_callback_ =
      std::bind(&Service<Request, Response>::HandleRequest, this,
                std::placeholders::_1, std::placeholders::_2);

  role.set_channel_name(request_channel_);
  channel_id = common::GlobalData::RegisterChannel(request_channel_);
  role.set_channel_id(channel_id);
  request_receiver_ = transport->CreateReceiver<Request>(
      role,
      [=](const std::shared_ptr<Request>& request,
          const transport::MessageInfo& message_info,
          const proto::RoleAttributes& reader_attr) {
        (void)reader_attr;
        auto task = [this, request, message_info]() {
          this->HandleRequest(request, message_info);
        };
        Enqueue(std::move(task));
      },
      proto::OptionalMode::RTPS);
  inited_ = true;
  thread_ = std::thread(&Service<Request, Response>::Process, this);
  if (request_receiver_ == nullptr) {
    AERROR << " Create request sub failed." << request_channel_;
    response_transmitter_.reset();
    return false;
  }
  return true;
}

template <typename Request, typename Response>
void Service<Request, Response>::HandleRequest(
    const std::shared_ptr<Request>& request,
    const transport::MessageInfo& message_info) {
  if (!IsInit()) {
    // LOG_DEBUG << "not inited error.";
    return;
  }
  ADEBUG << "handling request:" << request_channel_;
  std::lock_guard<std::mutex> lk(service_handle_request_mutex_);
  auto response = std::make_shared<Response>();
#ifdef ENABLE_TRACE
  TRACE_ENTRY_EXT(FrameworkTraceModule::SERVICECALLBACK, m_context);
  TRACE_RECEIVE_MSG(request);
  TRACE_CALLBACK_INFO(service_callback_);
  TRACE_SYS_PERF();
#endif
  service_callback_(request, response);
#ifdef ENABLE_TRACE
      TRACE_SEND_MSG(response);
#endif
  transport::MessageInfo msg_info(message_info);
  msg_info.set_sender_id(response_transmitter_->id());
  SendResponse(msg_info, response);
}

template <typename Request, typename Response>
void Service<Request, Response>::SendResponse(
    const transport::MessageInfo& message_info,
    const std::shared_ptr<Response>& response) {
  if (!IsInit()) {
    // LOG_DEBUG << "not inited error.";
    return;
  }
  // publish return value ?
  // LOG_DEBUG << "send response id:" << message_id.sequence_number;
  response_transmitter_->Transmit(response, message_info);
}

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_SERVICE_H_
