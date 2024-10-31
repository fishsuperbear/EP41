{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/BrakeRedInfo",
  "Namespace": "::ara::chass::report",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Uint8WithValid",
      "Padding": "2"
    },
    {
      "Name": "velocity",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Float32WithValid"
    },
    {
      "Name": "wheelSpeedFl",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/WheelSpeed"
    },
    {
      "Name": "wheelSpeedFr",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/WheelSpeed"
    },
    {
      "Name": "wheelSpeedRl",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/WheelSpeed"
    },
    {
      "Name": "wheelSpeedRr",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/WheelSpeed"
    },
    {
      "Name": "standstill",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Uint8WithValid"
    },
    {
      "Name": "espHwpActive",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "espHwpAvl",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driverOverrideAvl",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driverOverrideDomAv",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "2"
    }
  ],
  "IsPlane": "true"
}