/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: can publisher abstract class
 */

#ifndef CANSTACK_PUBLISHER_H
#define CANSTACK_PUBLISHER_H

#include "entity.h"
#include "cm/include/skeleton.h"

namespace hozon {
namespace netaos {
namespace canstack {

class Publisher : public Entity {
 public:
  // Publisher() = default;
  // Publisher(int domain, std::string topic);
  // ~Publisher() = default;
  virtual int Init() = 0;
  virtual void Pub() = 0;
  virtual int Stop() = 0;
// private:
//   std::shared_ptr<hozon::netaos::cm::Skeleton> skeleton_;
};

}  // namespace canstack
}
}  // namespace hozon
#endif  // CANSTACK_PUBLISHER_H
