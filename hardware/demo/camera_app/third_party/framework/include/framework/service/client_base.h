#ifndef CYBER_SERVICE_CLIENT_BASE_H_
#define CYBER_SERVICE_CLIENT_BASE_H_

#include <chrono>
#include <string>

#include "framework/common/macros.h"

namespace netaos {
namespace framework {

/**
 * @class ClientBase
 * @brief Base class of Client
 *
 */
class ClientBase {
 public:
  /**
   * @brief Construct a new Client Base object
   *
   * @param service_name the service we can request
   */
  explicit ClientBase(const std::string& service_name)
      : service_name_(service_name) {}
  virtual ~ClientBase() {}

  /**
   * @brief Destroy the Client
   */
  virtual void Destroy() = 0;

  /**
   * @brief Get the service name
   */
  const std::string& ServiceName() const { return service_name_; }

  /**
   * @brief Ensure whether there is any Service named `service_name_`
   */
  virtual bool ServiceIsReady() const = 0;

 protected:
  std::string service_name_;

  bool WaitForServiceNanoseconds(std::chrono::nanoseconds time_out) {
    bool has_service = false;
    auto step_duration = std::chrono::nanoseconds(5 * 1000 * 1000);
    while (time_out.count() > 0) {
      has_service = service_discovery::TopologyManager::Instance()
                        ->service_manager()
                        ->HasService(service_name_);
      if (!has_service) {
        std::this_thread::sleep_for(step_duration);
        time_out -= step_duration;
      } else {
        break;
      }
    }
    return has_service;
  }
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_CLIENT_BASE_H_
