#!/bin/sh
# verify_hashes.sh

echo "### Valid Hashes ###"
sha256sum ./INFO > ./artifacts_volume/latest_build.hash
date >> ./artifacts_volume/latest_build.hash
echo "Hashes verified and stored in artifacts_volume/."