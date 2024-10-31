#!/usr/bin/env bash

function downloadPkg() {
user='guest'
password='Hozon2022'
url='http://10.6.133.99:8082/artifactory'

thirdPartyPath='./third_party'

if [ ! -d ~/.jfrog ]; then
    mkdir -p ~/.jfrog && cp ./tools/jfrog/jfrog-cli.conf.v5 ~/.jfrog/
fi

currThirdParty=$(python3 -c "import sys, json;
with open('version.json','r') as f: print(json.load(f)['third_party'])")

echo "current third_party in version.json is $currThirdParty"
TPVersionInfo=''
if [ -f $thirdPartyPath/third_party/version.json ]; then
    TPVersionInfo=$(python3 -c "import sys, json;
with open('$thirdPartyPath/third_party/version.json','r') as f: print(json.load(f)['releaseVersion'])")
fi

if [[ "$TPVersionInfo" != "$currThirdParty" ]]; then
    echo "current third_party in third_party/version.json is $TPVersionInfo"
    ./tools/jfrog/jfrog rt dl --flat --user=$user --password=$password --url=$url EP40_MDC_TP/build/$currThirdParty*
    echo "tar -xf $currThirdParty.tar.gz "
    mkdir -p $thirdPartyPath/third_party && rm -rf $thirdPartyPath/third_party && tar -xf $currThirdParty.tar.gz -C $thirdPartyPath && rm -f $currThirdParty.tar.gz
fi

}

downloadPkg "$@"
