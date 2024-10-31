#ifndef CYBER_PARAMETER_PARAMETER_SERVER_H_
#define CYBER_PARAMETER_PARAMETER_SERVER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/proto/parameter.pb.h"

#include "framework/parameter/parameter.h"
#include "framework/service/service.h"

namespace netaos {
namespace framework {

class Node;

/**
 * @class ParameterServer
 * @brief Parameter Service is a very important function of auto-driving.
 * If you want to set a key-value, and hope other nodes to get the value,
 * Routing, sensor internal/external references are set by Parameter Service
 * ParameterServer can set a parameter, and then you can get/list
 * paramter(s) by start a ParameterClient to send responding request
 * @warning You should only have one ParameterServer works
 */
class ParameterServer {
 public:
  using Param = netaos::framework::proto::Param;
  using NodeName = netaos::framework::proto::NodeName;
  using ParamName = netaos::framework::proto::ParamName;
  using BoolResult = netaos::framework::proto::BoolResult;
  using Params = netaos::framework::proto::Params;
  /**
   * @brief Construct a new ParameterServer object
   *
   * @param node shared_ptr of the node handler
   */
  explicit ParameterServer(const std::shared_ptr<Node>& node);

  /**
   * @brief Set the Parameter object
   *
   * @param parmeter parameter to be set
   */
  void SetParameter(const Parameter& parmeter);

  /**
   * @brief Get the Parameter object
   *
   * @param parameter_name name of the parameer want to get
   * @param parameter pointer to store parameter want to get
   * @return true get parameter success
   * @return false parameter not exists
   */
  bool GetParameter(const std::string& parameter_name, Parameter* parameter);

  /**
   * @brief get all the parameters
   *
   * @param parameters result Paramter vector
   */
  void ListParameters(std::vector<Parameter>* parameters);

 private:
  std::shared_ptr<Node> node_;
  std::shared_ptr<Service<ParamName, Param>> get_parameter_service_;
  std::shared_ptr<Service<Param, BoolResult>> set_parameter_service_;
  std::shared_ptr<Service<NodeName, Params>> list_parameters_service_;

  std::mutex param_map_mutex_;
  std::unordered_map<std::string, Param> param_map_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_PARAMETER_PARAMETER_SERVER_H_
