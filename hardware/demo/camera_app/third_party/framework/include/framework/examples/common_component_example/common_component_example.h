#include <memory>

#include "framework/component/component.h"
#include "framework/examples/proto/examples.pb.h"

using netaos::framework::Component;
using netaos::framework::ComponentBase;
using netaos::framework::examples::proto::Driver;

class CommonComponentSample : public Component<Driver, Driver> {
 public:
  bool Init() override;
  bool Proc(const std::shared_ptr<Driver>& msg0,
            const std::shared_ptr<Driver>& msg1) override;
};
CYBER_REGISTER_COMPONENT(CommonComponentSample)
