key_name=home2
PASSWORD_FOR_NEXTCLOUD=______
WHO=______
mkdir -p ./TEMP
wget --user=fuqingxu  --password=$PASSWORD_FOR_NEXTCLOUD   http://cloud.$WHO.top:4080/remote.php/dav/files/fuqingxu/keys/$key_name.pub -O ./TEMP/_xkey
mkdir -p  ~/.ssh/
cat  ./TEMP/_xkey >>  ~/.ssh/authorized_keys 
cat  ~/.ssh/authorized_keys
