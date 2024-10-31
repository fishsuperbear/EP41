{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/VehicleChassisInfo",
  "Namespace": "::mdc::ssm",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "autoMode",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/AutonomyMode"
    },
    {
      "Name": "actualGear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Gear"
    },
    {
      "Name": "master",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/ActuatorStatus"
    },
    {
      "Name": "slave",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/ActuatorStatus",
      "Padding": "3"
    },
    {
      "Name": "steer",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/SteerInfo"
    },
    {
      "Name": "brake",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/BrakeInfo"
    },
    {
      "Name": "vcu",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/ThrottleInfo"
    },
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/GearInfo"
    },
    {
      "Name": "egoMotion",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/VehicleMotion"
    }
  ],
  "IsPlane": "true"
}