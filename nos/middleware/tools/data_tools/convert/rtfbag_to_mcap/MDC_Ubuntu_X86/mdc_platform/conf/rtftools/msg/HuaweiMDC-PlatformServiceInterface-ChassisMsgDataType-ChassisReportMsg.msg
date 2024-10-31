{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/ChassisReportMsg",
  "Namespace": "::ara::chassis",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "autoMode",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/AutonomyMode"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "steerAngle",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "frontWheelAngle",
      "Type": "/HuaweiMDC/StdCppDataType/Float"
    },
    {
      "Name": "actualGear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/Gear"
    },
    {
      "Name": "master",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/ActuatorStatus"
    },
    {
      "Name": "slave",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/ActuatorStatus"
    },
    {
      "Name": "steer",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/SteerInfo"
    },
    {
      "Name": "brake",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/BrakeInfo"
    },
    {
      "Name": "vcu",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/ThrottleInfo"
    },
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/GearInfo"
    },
    {
      "Name": "egoMotion",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/VehicleMotion"
    },
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/ChassisReportHeader"
    }
  ],
  "IsPlane": "false"
}