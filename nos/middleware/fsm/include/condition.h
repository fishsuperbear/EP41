#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "proto/statemachine/fsm_rule.pb.h"

namespace hozon {
namespace fsmcore {

struct Value;
class ConditionCore;

using ComputeCurValueFunc =
    std::function<Value(const std::vector<std::string>&)>;
using FindConditionCoreFunc =
    std::function<std::shared_ptr<ConditionCore>(const std::string& name)>;

/*****************************************************************************/
/* condition determine is comparing values, support following 4 value type   */
/*****************************************************************************/
enum class ValueType : std::uint32_t {
  BOOL = 0,  // 布尔
  STRING,    // 字符串
  INT,       // 整形（32位或64位）
  DOUBLE     // 浮点型
};

/*****************************************************************************/
/* condition determine is comparing values, support following 7 compare type */
/*****************************************************************************/
enum class OpType {
  LT,  // 小于
  LE,  // 小于等于
  EQ,  // 等于
  GE,  // 大于等于
  GT,  // 大于
  NE,  // 不等于
  IV   // 无效值
};

/*****************************************************************************/
/* only one val_xxx is using，others val_xxx only place holders              */
/*****************************************************************************/
struct Value {
  bool valid{false};
  bool val_bool{false};
  std::string val_string;
  int64_t val_int{0};
  double val_double{0.0};
  std::string to_string(const ValueType& value_type) {
    std::string return_string;

    if (!valid) {
      return_string += " invalid value";
      return return_string;
    }

    return_string += " valid value(type: ";

    switch (value_type) {
      case ValueType::BOOL:
        if (val_bool) {
          return_string += "bool): true";
        } else {
          return_string += "bool): false";
        }
        break;
      case ValueType::INT:
        return_string += "int): ";
        return_string += std::to_string(val_int);
        break;
      case ValueType::DOUBLE:
        return_string += "double): ";
        return_string += std::to_string(val_double);
        break;
      case ValueType::STRING:
        return_string += "string): ";
        return_string += val_string;
        break;
      default:
        break;
    }

    return return_string;
  }
};

#define VALUE_SATISFY(VAL, OP)                            \
  do {                                                    \
    if ((VAL) > 0) {                                      \
      if (((OP) == OpType::GT) || ((OP) == OpType::GE) || \
          ((OP) == OpType::NE)) {                         \
        return true;                                      \
      } else {                                            \
        return false;                                     \
      }                                                   \
    } else if ((VAL) < 0) {                               \
      if (((OP) == OpType::LT) || ((OP) == OpType::LE) || \
          ((OP) == OpType::NE)) {                         \
        return true;                                      \
      } else {                                            \
        return false;                                     \
      }                                                   \
    } else {                                              \
      if (((OP) == OpType::EQ) || ((OP) == OpType::GE) || \
          ((OP) == OpType::LE)) {                         \
        return true;                                      \
      } else {                                            \
        return false;                                     \
      }                                                   \
    }                                                     \
  } while (false)

/*****************************************************************************/
/* Used to determine condition is satisfy or not                             */
/*****************************************************************************/
class ConditionCore {
 public:
  ConditionCore(const std::string& name, ValueType type,
                const ComputeCurValueFunc& func);
  bool satisfy(const std::vector<std::string>& param, Value& cfg_value,
               const OpType& op);
  std::string to_string(Value& computed_value, Value& cfg_value, bool result);
  OpType parse_op_type_from_string(const std::string op_str);
  Value parse_value_from_string(const std::string value_str);
  std::string get_name() { return _core_name; }

 private:
  bool compare_value(const Value& cur, const Value& cfg, const OpType& op);
  Value compute_current_value(const std::vector<std::string>& input_param);

 private:
  std::unique_ptr<ComputeCurValueFunc> _compute_cur_val;
  ValueType _value_type;  // one ConditionCore has only one ValueType
  std::string _core_name;
};

/*****************************************************************************/
/* There are 3 kinds of Condition:                                           */
/*  1. Condition of Transit, only this kind has _transit_member to be true   */
/*  2. Leaf Condition, which has ZERO _sub_conditions but has _condition_core*/
/*  3. CompondCondition, but _transit_member is false                        */
/*****************************************************************************/
class Condition {
 public:
  Condition(const hozon::fsm_rule::FsmCondition& fsm_condition,
            FindConditionCoreFunc find_func, bool from_transit = false);
  bool satisfy();
  std::string to_string(const std::string& indent = "");

 private:
  Condition(const hozon::fsm_rule::Condition& condition,
            FindConditionCoreFunc find_func, bool from_transit = false);

 private:
  bool _transit_member;
  std::string _op_type_str;
  std::string _cfg_value_str;
  bool _is_and;
  OpType _op_type;
  Value _cfg_value;
  std::vector<std::string> _input_params;
  std::shared_ptr<ConditionCore> _condition_core;
  std::vector<std::shared_ptr<Condition>> _sub_conditions;
};

}  // namespace fsmcore
}  // namespace hozon