{
  "ShortName": "/HuaweiMDC/FunctionalSoftware/NavigationResultMsgDataType/NavigationResultStruct",
  "Namespace": "::ara::navigation",
  "Type": "STRUCTURE",
  "Subelements": [
    {
      "Name": "header",
      "Type": "/HuaweiMDC/FunctionalSoftware/NavigationResultMsgDataType/Header"
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
      "Name": "isNewRouting",
      "Type": "/HuaweiMDC/StdCppBasicDataType/bool"
    },
    {
      "Name": "blacklistRoads",
      "Type": "/HuaweiMDC/ApplicationDataType/StringVector"
    },
    {
      "Name": "blacklistLaneSegments",
      "Type": "/HuaweiMDC/ApplicationComplexDataType/NavigationDataType/LanePieceVector"
    }
  ],
  "IsPlane": "false"
}