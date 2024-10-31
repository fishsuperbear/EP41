{
  "ShortName": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanFdSendMsgDataType/CanFdBusDataParam",
  "Namespace": "::mdc::mdccan",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "seq",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "sendType",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
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
      "Type": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanFdSendMsgDataType/CanFdSendData"
    },
    {
      "Name": "reserved",
      "Type": "/HuaweiMDC/MdcPlatformServices/MsgDataType/CanComMsgType/ReservedData"
    },
    {
      "Name": "checkSum",
      "Type": "/HuaweiMDC/StdCppDataType/UInt32"
    }
  ],
  "IsPlane": "false"
}