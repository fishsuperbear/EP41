/*
 * Copyright (c) Hozon Auto Co., Ltd. 2021-2021. All rights reserved.
 * Description: entity abstract class
 */

#ifndef CAN_ENTITY_H
#define CAN_ENTITY_H

namespace hozon {
namespace netaos {
namespace canstack {

class Entity {
 public:
  Entity() = default;
  ~Entity() = default;

  virtual int Init() = 0;
  virtual int Stop() = 0;
};

}  // namespace canstack
}
}  // namespace hozon
#endif  // CAN_ENTITY_H
