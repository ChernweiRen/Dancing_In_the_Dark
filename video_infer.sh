name="14LUX"
video_suffix='avi'

suffix="_result"
path="/mnt/raid0/rcw/lowlight_test"

H=1080
W=1920

ckpt="checkpoint/model_best.pth.tar"

ffmpeg -i $path/$name.$video_suffix -an -c:v rawvideo -pixel_format yuv420p $path/$name.yuv

mkdir $path/$name
ffmpeg -y -s $W\x$H -pix_fmt yuv420p -i $path/$name.yuv $path/$name/%d.png 

python convert.py --path $path/$name

mkdir $path/$name$suffix

# max=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 $path/$name.avi)

# for i in $(seq 1 $max)
# do   
# echo $i
python inference_folder.py \
    --arch_type GC4653 \
    --input $path/$name/npy/ \
    --resume $ckpt \
    --result_dir $path/$name$suffix/ \
    --backbone UNet
    --gpu 0
# done

ffmpeg -f image2 -i $path/$name$suffix/%d_out.png -crf 17 -b:v 4M -pix_fmt yuv420p $name$suffix.MP4