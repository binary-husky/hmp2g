

# Get Unreal-MAP Binary Client (Win & Linux)

- Method 1 (Will run automatically if U-MAP is called but cannot find executable): 
``` python 
from MISSION.uhmap.auto_download import download_client_binary_on_platform
download_client_binary_on_platform(
    desired_path="./UnrealHmapBinary/Version3.5/LinuxNoEditor/UHMP.sh", 
    # desired_path="./UnrealHmapBinary/Version3.5/LinuxNoEditor/UHMP.exe", 
    desired_version="3.5", 
    is_render_client=True,
    platform="Linux",
    # platform="Windows",
)
```

- Method 2 (manual): download uhmap file manifest (a json file)
```
https://ageasga-my.sharepoint.com/:u:/g/personal/fuqingxu_yiteam_tech/EVmCQMSUWV5MgREWaxiz_GoBalBRV3DWBU3ToSJ5OTQaLQ?e=I8yjl9
```
Open this json file, choose the version and platform you want, download and unzip it.


- Method 3 (Compile from source): 
If you want to build new multi-agent environment, welcome to our U-MAP project in following resp!
In U-MAP, you can use all the capability of Enreal Engine (blueprints, behavior tree, physics engine, AI navigation, 3D models/animations and rich plugin resources, etc) to build 
elegant (but also computational efficient) and magnificent (but also experimentally reproducible) MARL tasks.
```
https://github.com/binary-husky/unreal-map
```
