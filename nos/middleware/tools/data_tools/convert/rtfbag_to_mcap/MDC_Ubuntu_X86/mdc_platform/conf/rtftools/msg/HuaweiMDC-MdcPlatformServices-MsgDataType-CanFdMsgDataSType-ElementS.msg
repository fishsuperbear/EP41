{
  "ShortName": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanFdMsgDataSType/ElementS",
  "Namespace": "::mdc::mdccan",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "canIdType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "validLen",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "timeStamp",
      "Type": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanComMsgType/Systemtime"
    },
    {
      "Name": "canId",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "data",
      "Type": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanFdMsgDataSType/CanFdDataS"
    },
    {
      "Name": "checkSum",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    },
    {
      "Name": "reserved",
      "Type": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanComMsgType/ReservedData"
    }
  ],
  "IsPlane": "false"
}