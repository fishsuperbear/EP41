{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/GearInfo",
  "Namespace": "::ara::chassis9000",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Gear"
    },
    {
      "Name": "gearLever",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Gear"
    },
    {
      "Name": "gearShiftStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/Chassis9000MsgDataType/Uint8WithValid",
      "Padding": "3"
    },
    {
      "Name": "faultCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    },
    {
      "Name": "gearCmd",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "gearWorkMode",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8",
      "Padding": "2"
    }
  ],
  "IsPlane": "true"
}