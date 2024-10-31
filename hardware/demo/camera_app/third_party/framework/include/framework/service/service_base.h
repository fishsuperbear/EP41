#ifndef CYBER_SERVICE_SERVICE_BASE_H_
#define CYBER_SERVICE_SERVICE_BASE_H_

#include <string>

namespace netaos {
namespace framework {

/**
 * @class ServiceBase
 * @brief Base class for Service
 *
 */
class ServiceBase {
 public:
  /**
   * @brief Construct a new Service Base object
   *
   * @param service_name name of this Service
   */
  explicit ServiceBase(const std::string& service_name)
      : service_name_(service_name) {}

  virtual ~ServiceBase() {}

  virtual void destroy() = 0;

  /**
   * @brief Get the service name
   */
  const std::string& service_name() const { return service_name_; }

 protected:
  std::string service_name_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_SERVICE_SERVICE_BASE_H_
