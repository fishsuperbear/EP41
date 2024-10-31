#include "condition.h"

#include "fsm.h"
#include "fsm_utils.h"

namespace hozon {
namespace fsmcore {

///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************/
/* 构造函数 */
/*****************************************************************************/
ConditionCore::ConditionCore(const std::string& name, ValueType type,
                             const ComputeCurValueFunc& func)
    : _value_type(type), _core_name(name) {
  _compute_cur_val = std::make_unique<ComputeCurValueFunc>(func);
}

/*****************************************************************************/
/* 根据输入的字符串，解析出来操作符，比如 "lt"\"eq"*/
/*****************************************************************************/
OpType ConditionCore::parse_op_type_from_string(const std::string op_str) {
  static std::unordered_map<std::string, OpType> s_op{
      {"lt", OpType::LT}, {"gt", OpType::GT}, {"eq", OpType::EQ},
      {"ne", OpType::NE}, {"le", OpType::LE}, {"ge", OpType::GE},
      {"iv", OpType::IV}};

  OpType op = s_op[op_str];

  if (_value_type == ValueType::BOOL) {
    if ((op != OpType::EQ) && (op != OpType::NE) && (op != OpType::IV)) {
      throw "bool value type should has only EQ or NE or IV operator type, but now : " + op_str;
    }
  }

  return op;
}

/*****************************************************************************/
/* 根据输入的字符串，解析出来用 Value 数据结构描述的值，比如 "5"\"3.3" */
/*****************************************************************************/
Value ConditionCore::parse_value_from_string(const std::string value_str) {
  Value value;
  value.valid = true;

  switch (_value_type) {
    case ValueType::STRING:
      value.val_string = value_str;
      break;
    case ValueType::INT:
      value.val_int = std::stoull(value_str);
      break;
    case ValueType::DOUBLE:
      value.val_double = std::stod(value_str);
      break;
    case ValueType::BOOL:
      if (value_str == "true") {
        value.val_bool = true;
      } else if (value_str == "false") {
        value.val_bool = false;
      } else {
        throw "parse bool value string err";
      }
      break;
    default:
      break;
  }

  return value;
}

/*****************************************************************************/
/* 基于输入参数、配置值以及两者之间运算逻辑，看能否条件成立 */
/*****************************************************************************/
bool ConditionCore::satisfy(const std::vector<std::string>& param,
                            Value& cfg_value, const OpType& op) {
  FSMCORE_LOG_INFO << "Condition " << _core_name << " input " << param.size()
                   << " parameters: ";
  uint32_t param_counter = 0;

  for (auto&& param_str : param) {
    FSMCORE_LOG_INFO << "\tParameter " << (param_counter++) << ": "
                     << param_str;
  }
  auto cur_value = compute_current_value(param);
  bool ret;
  if (cur_value.valid) {
    if (op != OpType::IV) {
      // most of situations
      ret = compare_value(cur_value, cfg_value, op);
    } else {
      ret = false;
    }
  } else {
    if (op == OpType::IV) {
      ret = true;
    } else {
      ret = false;
    }
  }

  FSMCORE_LOG_INFO << to_string(cur_value, cfg_value, ret);
  return ret;
}

std::string ConditionCore::to_string(Value& computed_value, Value& cfg_value,
                                     bool result) {
  std::string return_string;
  return_string += "Comparing";
  return_string += computed_value.to_string(_value_type);
  return_string += " with expected";
  return_string += cfg_value.to_string(_value_type);
  return_string += ", result is ";

  auto ret_string = (result ? "Satisfied." : "NOT satisfy.");
  return_string += ret_string;

  return return_string;
}

/*****************************************************************************/
/* 根据入参计算当前值。如果未绑定处理函数，返回无效值 */
/*****************************************************************************/
Value ConditionCore::compute_current_value(
    const std::vector<std::string>& input_param) {
  if (!NULL_IPTR(_compute_cur_val)) {
    auto func = _compute_cur_val.get();
    return (*func)(input_param);
  } else {
    Value val;
    val.valid = false;
    return val;
  }
}

/*****************************************************************************/
/* 基于输入值、配置值以及两者之间运算逻辑，看能否条件成立 */
/*****************************************************************************/
bool ConditionCore::compare_value(const Value& cur, const Value& cfg,
                                  const OpType& op) {
  if (_value_type == ValueType::BOOL) {
    auto equal = (cur.val_bool == cfg.val_bool);
    if (op == OpType::NE) {
      equal = !equal;
    }

    return equal;
  } else {
    if (_value_type == ValueType::STRING) {
      auto string_val = cur.val_string.compare(cfg.val_string);
      VALUE_SATISFY(string_val, op);
    } else if (_value_type == ValueType::INT) {
      auto int_val = cur.val_int - cfg.val_int;
      VALUE_SATISFY(int_val, op);
    } else if (_value_type == ValueType::DOUBLE) {
      auto double_val = cur.val_double - cfg.val_double;
      VALUE_SATISFY(double_val, op);
    } else {
      return false;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////

/*****************************************************************************/
/* 私有构造函数，给普通构造函数调用 */
/*****************************************************************************/
Condition::Condition(const hozon::fsm_rule::Condition& condition,
                     FindConditionCoreFunc find_func, bool from_transit) {
  _op_type_str = "";
  _cfg_value_str = "";
  _transit_member = from_transit;

  if (condition.has_op()) {
    _op_type_str = condition.op();
  }

  if (condition.has_value()) {
    _cfg_value_str = condition.value();
  }

  for (auto param : condition.params()) {
    _input_params.push_back(param);
  }

  auto core_name = condition.name();
  _condition_core = find_func(core_name);
  if (NULL_IPTR(_condition_core)) {
    FSMCORE_LOG_FATAL << "Cannot find condition core with core name: "
                      << core_name;
    exit(-1);
  }

  _op_type = _condition_core->parse_op_type_from_string(_op_type_str);

  if (_op_type != OpType::IV) {
    _cfg_value = _condition_core->parse_value_from_string(_cfg_value_str);
  }
}

/*****************************************************************************/
/* 普通构造函数，调用私有构造函数 */
/*****************************************************************************/
Condition::Condition(const hozon::fsm_rule::FsmCondition& fsm_condition,
                     FindConditionCoreFunc find_func, bool from_transit)
    : _transit_member(from_transit) {
  _is_and = fsm_condition.is_and();

  // leaf condition does only has this
  for (int ii = 0; ii < fsm_condition.conditions_size(); ++ii) {
    _sub_conditions.emplace_back(std::shared_ptr<Condition>(
        new Condition(fsm_condition.conditions(ii), find_func, true)));
  }

  for (int ii = 0; ii < fsm_condition.fsm_conditions_size(); ++ii) {
    _sub_conditions.emplace_back(std::make_shared<Condition>(
        fsm_condition.fsm_conditions(ii), find_func));
  }
}

/*****************************************************************************/
/* 本条件是否满足 */
/*****************************************************************************/
bool Condition::satisfy() {
  // 纯条件，非复合条件
  if (nullptr != _condition_core.get()) {
    auto ret = _condition_core->satisfy(_input_params, _cfg_value, _op_type);
    if (ret) {
      FSMCORE_LOG_WARN << "Check [" << to_string() << "] is Satisfied.";
    } else {
      FSMCORE_LOG_ERROR << "Check [" << to_string() << "] is NOT Satisfy.";
    }

    return ret;
  }

  // 以下为复合条件，递归调用，最终会调用上面几行代码
  // 空条件，恒成立
  if (_sub_conditions.empty()) {
    FSMCORE_LOG_ERROR << "NULL compond condition: "
                      << _condition_core->get_name() << ", treated as satify.";
    return true;
  }

  // 以下为非空复合条件，需要判断之间的与或关系
  // 与条件，只要有一个不成立，都不成立
  if (_is_and) {
    for (auto&& condition : _sub_conditions) {
      if (!condition->satisfy()) {
        return false;
      }
    }
    return true;
  }

  // 或条件，只要有一个成立，都成立
  for (auto&& condition : _sub_conditions) {
    if (condition->satisfy()) {
      return true;
    }
  }
  return false;
}

/*****************************************************************************/
/* 将当前的条件打印出来 */
/*****************************************************************************/
std::string Condition::to_string(const std::string& indent) {
  std::string return_str;
  if (!NULL_IPTR(_condition_core)) {
    return_str += indent + _condition_core->get_name() + " " + _op_type_str +
                  " " + _cfg_value_str;
    if (_transit_member) {
      return_str += " (direct of transit)";
    } else {
      return_str += " (indirect of transit)";
    }
  } else {
    for (auto&& sub : _sub_conditions) {
      if (return_str.length() > 0) {
        return_str += indent + (_is_and ? "&&" : "||") + "\n";
      }
      return_str += sub->to_string(indent + " ");
    }
  }
  return return_str;
}

}  // namespace fsmcore
}  // namespace hozon
