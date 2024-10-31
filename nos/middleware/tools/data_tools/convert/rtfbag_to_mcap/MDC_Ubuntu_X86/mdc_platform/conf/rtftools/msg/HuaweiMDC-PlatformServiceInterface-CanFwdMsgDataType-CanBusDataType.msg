{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/CanFwdMsgDataType/CanBusDataType",
  "Namespace": "::ara::intfcanfwd",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/CanFwdMsgDataType/Header"
    },
    {
      "Name": "canId",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "canType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "data",
      "Type": "/HuaweiMDC/PlatformServiceInterface/CanFwdMsgDataType/Uint8Vector"
    }
  ],
  "IsPlane": "false"
}