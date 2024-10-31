{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/GearInfo",
  "Namespace": "::mdc::ssm",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "gear",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Gear"
    },
    {
      "Name": "gearLever",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Gear"
    },
    {
      "Name": "gearShiftStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "driverOverride",
      "Type": "/HuaweiMDC/PlatformServiceInterface/VehicleInfoMsgDataType/Uint8WithValid",
      "Padding": "3"
    },
    {
      "Name": "faultCode",
      "Type": "/HuaweiMDC/StdCppDataType/Int32"
    }
  ],
  "IsPlane": "true"
}