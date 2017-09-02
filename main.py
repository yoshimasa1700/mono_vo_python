import argparse


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('--calibfile')

    return parser.parse_args()


class CameraParameters():
    def __init__(self, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


def loadCameraParameters(calibfile):
    with open(calibfile, 'r') as f:
        line = f.readline()
        print(line)


def main():
    options = parseArguments()

    image_path = options.path
    if options.calibfile:
        camParam = loadCameraParameters(options.calibfile)

    print(camParam)

if __name__ == "__main__":
    main()
