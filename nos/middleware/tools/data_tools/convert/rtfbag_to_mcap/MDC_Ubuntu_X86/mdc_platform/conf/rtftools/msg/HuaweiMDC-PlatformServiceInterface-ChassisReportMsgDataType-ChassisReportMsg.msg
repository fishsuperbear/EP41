{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/ChassisReportMsg",
  "Namespace": "::ara::chass::report",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "autoMode",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/AutonomyMode"
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
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Gear"
    },
    {
      "Name": "master",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/ActuatorStatus"
    },
    {
      "Name": "slave",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/ActuatorStatus"
    },
    {
      "Name": "steer",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/SteerInfo"
    },
    {
      "Name": "brake",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/BrakeInfo"
    },
    {
      "Name": "vcu",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/ThrottleInfo"
    },
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/GearInfo"
    },
    {
      "Name": "egoMotion",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/VehicleMotion"
    },
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/ChassisReportHeader"
    }
  ],
  "IsPlane": "false"
}