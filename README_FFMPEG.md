Greyscale

ffmpeg -i teste.mp4 -vf hue=s=0 output.mp4

Transformar em imagem pra ler

ffmpeg -i output.mp4 thumb%04d.png -hide_banner

colocar na image_x

