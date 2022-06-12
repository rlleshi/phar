#!/bin/bash

# update bashrc
ZZROOT="${HOME}/app"

echo "export PATH=${ZZROOT}/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${ZZROOT}/lib:${ZZROOT}/lib64:${LD_LIBRARY_PATH}" >> ~/.bashrc
echo "export OpenCV_DIR=${ZZROOT}" >> ~/.bashrc

echo "export BOOST_ROOT=${ZZROOT}" >> ~/.bashrc

# finally source it
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
fi
