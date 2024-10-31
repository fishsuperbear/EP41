#ifndef CYBER_PARAMETER_PARAMETER_SERVICE_NAMES_H_
#define CYBER_PARAMETER_PARAMETER_SERVICE_NAMES_H_

#include <string>

namespace netaos {
namespace framework {

constexpr auto SERVICE_NAME_DELIMITER = "/";
constexpr auto GET_PARAMETER_SERVICE_NAME = "get_parameter";
constexpr auto SET_PARAMETER_SERVICE_NAME = "set_parameter";
constexpr auto LIST_PARAMETERS_SERVICE_NAME = "list_parameters";

static inline std::string FixParameterServiceName(const std::string& node_name,
                                                  const char* service_name) {
  return node_name + std::string(SERVICE_NAME_DELIMITER) +
         std::string(service_name);
}

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_PARAMETER_PARAMETER_SERVICE_NAMES_H_
