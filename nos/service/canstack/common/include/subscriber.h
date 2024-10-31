/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: can subscriber abstract class
 */

#ifndef CANSTACK_SUBSCRIBER_H
#define CANSTACK_SUBSCRIBER_H

#include "entity.h"

namespace hozon {
namespace netaos {
namespace canstack {

class Subscriber : public Entity {
 public:
  virtual int Init() = 0;
  virtual void Sub() = 0;
  virtual int Stop() = 0;
};

}  // namespace canstack  
}
}  // namespace hozon
#endif  // CANSTACK_SUBSCRIBER_H
