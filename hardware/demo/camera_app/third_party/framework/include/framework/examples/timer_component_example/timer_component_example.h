#include <memory>

#include "framework/class_loader/class_loader.h"
#include "framework/component/component.h"
#include "framework/component/timer_component.h"
#include "framework/examples/proto/examples.pb.h"

using netaos::framework::Component;
using netaos::framework::ComponentBase;
using netaos::framework::TimerComponent;
using netaos::framework::Writer;
using netaos::framework::examples::proto::Driver;

class TimerComponentSample : public TimerComponent {
 public:
  bool Init() override;
  bool Proc() override;

 private:
  std::shared_ptr<Writer<Driver>> driver_writer_ = nullptr;
};
CYBER_REGISTER_COMPONENT(TimerComponentSample)
