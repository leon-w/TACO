# get the first 20s of a video and annotate it
function annotate {
    ffmpeg  -i ~/../data/trash_detection/videos/$@.mp4 -to 20 -c copy $@_short.mp4
    python3 annotate.py --model=mask_rcnn_taco_0100.h5 --input=$@_short.mp4 --output=annotated_$@_short.mp4
}