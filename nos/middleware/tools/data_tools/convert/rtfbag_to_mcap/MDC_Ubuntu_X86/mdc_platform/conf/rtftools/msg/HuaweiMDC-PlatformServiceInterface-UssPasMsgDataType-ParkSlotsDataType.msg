{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/UssPasMsgDataType/ParkSlotsDataType",
  "Namespace": "::mdc::pasparkslots",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssComMsgDataType/Header"
    },
    {
      "Name": "vehicleInfoMsg",
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssPasMsgDataType/VehicleInfo"
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
      "Type": "/HuaweiMDC/PlatformServiceInterface/UssPasMsgDataType/ParkSlotsVec"
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