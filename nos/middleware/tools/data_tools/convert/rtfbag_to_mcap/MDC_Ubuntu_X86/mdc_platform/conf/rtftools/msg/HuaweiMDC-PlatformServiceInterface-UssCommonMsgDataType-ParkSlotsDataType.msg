{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssCommonMsgDataType/ParkSlotsDataType",
  "Namespace": "::mdc::commonparkslots",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Header"
    },
    {
      "Name": "msgRollingCounterLeft",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "msgRollingCounterRight",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "parkSlotSystemStatus",
      "Type": "/HuaweiMDC/StdCppDataType/UInt16"
    },
    {
      "Name": "parkSlotsList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssCommonMsgDataType/ParkSlotsVec"
    },
    {
      "Name": "reserveMsg1",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "reserveMsg2",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    }
  ],
  "IsPlane": "false"
}