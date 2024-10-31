{
  "ShortName": "/HuaweiMDC/PlatformServiceInterface/RadarTrackMsgDataType/RadarTrackArray",
  "Namespace": "::ara::rtrack",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/Header"
    },
    {
      "Name": "version",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/Version"
    },
    {
      "Name": "sensorId",
      "Type": "/HuaweiMDC/StdCppDataType/UInt8"
    },
    {
      "Name": "radarState",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/RadarState"
    },
    {
      "Name": "trackList",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarTrackMsgDataType/TrackList"
    },
    {
      "Name": "guardnailInfo",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/GuardnailInfo"
    },
    {
      "Name": "timeStampInfo",
      "Type": "/HuaweiMDC/PlatformServiceInterface/RadarMsgDataType/TimeStampInfo"
    }
  ],
  "IsPlane": "false"
}