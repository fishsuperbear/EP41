{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/GearInfo",
  "Namespace": "::ara::chassis",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/Gear"
    },
    {
      "Name": "gearLever",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/Gear"
    },
    {
      "Name": "gearShiftStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/ChassisMsgDataType/Uint8WithValid",
      "Padding": "3"
    },
    {
      "Name": "faultCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    }
  ],
  "IsPlane": "true"
}