mkdir -p ./TEMP
PASSWORD_FOR_NEXTCLOUD=______
WHO=______
wget --user=fuqingxu  --password=$PASSWORD_FOR_NEXTCLOUD   http://cloud.$WHO.top:4080/remote.php/dav/files/fuqingxu/keys/v2ray.zip -O ./TEMP/v2ray.zip
cd ./TEMP
unzip v2ray.zip
cd ./v2ray
chmod +x ./v2ray
./v2ray


sudo nano /etc/proxychains.conf
>> socks5 localhost 1080

rm -r ./TEMP
