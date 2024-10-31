#ifndef CYBER_COMPONENT_TIMER_COMPONENT_H_
#define CYBER_COMPONENT_TIMER_COMPONENT_H_

#include <memory>

#include "framework/component/component_base.h"

namespace netaos {
namespace framework {

class Timer;

/**
 * @brief .
 * TimerComponent is a timer component. Your component can inherit from
 * Component, and implement Init() & Proc(), They are called by the netaos framework.
 */
class TimerComponent : public ComponentBase {
 public:
  TimerComponent();
  ~TimerComponent() override;

  /**
   * @brief init the component by protobuf object.
   *
   * @param config which is define in 'framework/proto/component_conf.proto'
   *
   * @return returns true if successful, otherwise returns false
   */
  bool Initialize(const TimerComponentConfig& config) override;
  void Clear() override;
  bool Process();
  uint64_t GetInterval() const;

 private:
  /**
   * @brief The Proc logic of the component, which called by the netaos framework.
   *
   * @return returns true if successful, otherwise returns false
   */
  virtual bool Proc() = 0;

  uint64_t interval_ = 0;
  std::unique_ptr<Timer> timer_;
};

}  // namespace framework
}  // namespace netaos

#endif  // CYBER_COMPONENT_TIMER_COMPONENT_H_
