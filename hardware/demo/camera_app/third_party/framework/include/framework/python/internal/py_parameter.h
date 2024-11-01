#ifndef CYBER_PYTHON_INTERNAL_PY_PARAMETER_H_
#define CYBER_PYTHON_INTERNAL_PY_PARAMETER_H_

#include <unistd.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "framework/cyber.h"
#include "framework/init.h"
#include "framework/parameter/parameter.h"
#include "framework/parameter/parameter_client.h"
#include "framework/parameter/parameter_server.h"

namespace netaos {
namespace framework {

class PyParameter {
 public:
  PyParameter() {}
  explicit PyParameter(Parameter* param) : parameter_(*param) {}

  PyParameter(const std::string& name, const int64_t int_value)
      : parameter_(name, int_value) {}
  PyParameter(const std::string& name, const double double_value)
      : parameter_(name, double_value) {}
  PyParameter(const std::string& name, const std::string& string_value)
      : parameter_(name, string_value) {}
  PyParameter(const std::string& name, const std::string& msg_str,
              const std::string& full_name, const std::string& proto_desc)
      : parameter_(name, msg_str, full_name, proto_desc) {}

  uint type() { return parameter_.Type(); }

  std::string type_name() { return parameter_.TypeName(); }

  std::string descriptor() { return parameter_.Descriptor(); }

  std::string name() { return parameter_.Name(); }

  int64_t as_int64() { return parameter_.AsInt64(); }
  double as_double() { return parameter_.AsDouble(); }
  std::string as_string() { return parameter_.AsString(); }
  std::string debug_string() { return parameter_.DebugString(); }

  Parameter& get_param() { return parameter_; }

 private:
  Parameter parameter_;
};

class PyParameterClient {
 public:
  PyParameterClient(const std::shared_ptr<Node>& node,
                    const std::string& service_node_name)
      : parameter_clt_(node, service_node_name) {}

  bool set_parameter(const Parameter& parameter) {
    return parameter_clt_.SetParameter(parameter);
  }
  bool get_parameter(const std::string& param_name, Parameter* parameter) {
    return parameter_clt_.GetParameter(param_name, parameter);
  }
  bool list_parameters(std::vector<Parameter>* parameters) {
    return parameter_clt_.ListParameters(parameters);
  }

 private:
  ParameterClient parameter_clt_;
};

class PyParameterServer {
 public:
  explicit PyParameterServer(const std::shared_ptr<Node>& node)
      : parameter_srv_(node) {}

  void set_parameter(const Parameter& parameter) {
    parameter_srv_.SetParameter(parameter);
  }
  bool get_parameter(const std::string& param_name, Parameter* parameter) {
    return parameter_srv_.GetParameter(param_name, parameter);
  }
  void list_parameters(std::vector<Parameter>* parameters) {
    parameter_srv_.ListParameters(parameters);
  }

 private:
  ParameterServer parameter_srv_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_PYTHON_INTERNAL_PY_PARAMETER_H_
