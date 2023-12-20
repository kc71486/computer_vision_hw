import hwutil

def main():
    vw = hwutil.VideoWrapper()
    video = vw.read()
    fps = vw.get_fps()
    
    
    print("end")

# ffmpeg -i ./traffic.mp4 ./frames/traffic_%03d.png
if __name__ == "__main__":
    main()
