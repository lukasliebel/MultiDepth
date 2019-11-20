docker run \
--runtime=nvidia \
--rm \
-d \
--name multidepth \
-it \
--ipc=host \
-v [path to multidepth code]:/root/code \
-v [path to kitti depth maps]:/root/data/kitti/depth \
-v [path to kitti rgb images]:/root/data/kitti/rgb \
-v [path to an external log dir]:/root/logs \
lukasliebel/multidepth
