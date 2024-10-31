{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/NavigationInfoMsgDataType/NavigationInfo",
  "Namespace": "::adsfi",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/CommonDataTypes/CommonHeader"
    },
    {
      "Name": "roadPieces",
      "Type": "/HuaweiMDC/ApplicationComplexDataType/NavigationDataType/RoadPieceVector"
    },
    {
      "Name": "selectPoints",
      "Type": "/HuaweiMDC/ApplicationComplexDataType/NavigationDataType/NaviRoadPointVector"
    },
    {
      "Name": "mapVersion",
      "Type": "/HuaweiMDC/StdCppDataType/String"
    },
    {
      "Name": "isReRouted",
      "Type": "/HuaweiMDC/StdCppDataType/Boolean"
    },
    {
      "Name": "blacklistRoads",
      "Type": "/HuaweiMDC/ApplicationDataType/StringVector"
    },
    {
      "Name": "blacklistLanePieces",
      "Type": "/HuaweiMDC/ApplicationComplexDataType/NavigationDataType/LanePieceVector"
    }
  ],
  "IsPlane": "false"
}