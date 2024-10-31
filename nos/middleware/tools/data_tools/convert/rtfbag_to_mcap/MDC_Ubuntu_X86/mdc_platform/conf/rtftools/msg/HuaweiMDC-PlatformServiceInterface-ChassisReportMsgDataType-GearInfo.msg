{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/GearInfo",
  "Namespace": "::ara::chass::report",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Gear"
    },
    {
      "Name": "gearLever",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Gear"
    },
    {
      "Name": "gearShiftStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisReportMsgDataType/Uint8WithValid",
      "Padding": "3"
    },
    {
      "Name": "faultCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    }
  ],
  "IsPlane": "true"
}