#ifndef CYBER_PARAMETER_PARAMETER_CLIENT_H_
#define CYBER_PARAMETER_PARAMETER_CLIENT_H_

#include <memory>
#include <string>
#include <vector>

#include "framework/proto/parameter.pb.h"

#include "framework/parameter/parameter.h"
#include "framework/service/client.h"

namespace netaos {
namespace framework {

class Node;

/**
 * @class ParameterClient
 * @brief Parameter Client is used to set/get/list parameter(s)
 * by sending a request to ParameterServer
 */
class ParameterClient {
 public:
  using Param = netaos::framework::proto::Param;
  using NodeName = netaos::framework::proto::NodeName;
  using ParamName = netaos::framework::proto::ParamName;
  using BoolResult = netaos::framework::proto::BoolResult;
  using Params = netaos::framework::proto::Params;
  using GetParameterClient = Client<ParamName, Param>;
  using SetParameterClient = Client<Param, BoolResult>;
  using ListParametersClient = Client<NodeName, Params>;
  /**
   * @brief Construct a new ParameterClient object
   *
   * @param node shared_ptr of the node handler
   * @param service_node_name node name which provide a param services
   */
  ParameterClient(const std::shared_ptr<Node>& node,
                  const std::string& service_node_name);

  /**
   * @brief Get the Parameter object
   *
   * @param param_name
   * @param parameter the pointer to store
   * @return true
   * @return false call service fail or timeout
   */
  bool GetParameter(const std::string& param_name, Parameter* parameter);

  /**
   * @brief Set the Parameter object
   *
   * @param parameter parameter to be set
   * @return true set parameter succues
   * @return false 1. call service timeout
   *               2. parameter not exists
   *               The corresponding log will be recorded at the same time
   */
  bool SetParameter(const Parameter& parameter);

  /**
   * @brief Get all the Parameter objects
   *
   * @param parameters pointer of vector to store all the parameters
   * @return true
   * @return false call service fail or timeout
   */
  bool ListParameters(std::vector<Parameter>* parameters);

 private:
  std::shared_ptr<Node> node_;
  std::shared_ptr<GetParameterClient> get_parameter_client_;
  std::shared_ptr<SetParameterClient> set_parameter_client_;
  std::shared_ptr<ListParametersClient> list_parameters_client_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_PARAMETER_PARAMETER_CLIENT_H_
