apt-get -y update
apt-get install python3-pip
pip3 install virtualenv
virtualenv roberta --python=python3
cd ./roberta
source bin/activate
cd ..//
pip3 install --upgrade pip
pip3 install torch==1.8.1
pip3 install -r requirements.txt

apt-get install git
apt-get install wget
apt-get install nano
apt-get install unzip
git clone https://github.com/NVIDIA/apex
cd apex
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ../
wget -O data.zip "https://drive.google.com/uc?export=download&id=1AF-U0jjud1eJaKmWVxNBLvQdpxp6LEj3"
unzip data.zip
rm data.zip
wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id=1Ob-CVMlfBBRhcBpG1-iaPP_g9SZYjW2s' -O- \
     | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O models.zip \
     'https://docs.google.com/uc?export=download&id=1Ob-CVMlfBBRhcBpG1-iaPP_g9SZYjW2s&confirm='$(<confirm.txt)
unzip models.zip
rm models.zip
rm confirm.txt
rm cookies.txt