import numpy as np
import cv2


def pull_vidcap(vidcap, size=64, time_slice_factor=1):

    X = list()
    t = 0
    while(vidcap.isOpened()):
        try:
            ret, frame = vidcap.read()
            if (t % time_slice_factor) == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (size, size))
                X.append(frame)
        except:
            break
    return np.array(X)

def main(path_to_file):
#     file = path_to_file + 'Breakfast_Nonsocial.mov'
#     vidcap = cv2.VideoCapture(file)

#     # count the frames -- should be ~30/second
#     n_frames = 0
#     while (vidcap.isOpened()):
#         try:
#             ret, frame = vidcap.read()
#             n_frames += 1
#         except:
#             break

#     print("There are %d frames in 185s, or %f frames/second" % (n_frames, n_frames / 185.))

    X, y = [], []

    files = 'Breakfast_Nonsocial.mov Laundry_Nonsocial.mov legosShort.mov Library_Nonsocial.mov Party_Nonsocial.mov wl_dv.mov'.split()
    for ii, file in enumerate(files):
        vidcap = cv2.VideoCapture(path_to_file + file)
        _X = pull_vidcap(vidcap)
        X.append(_X)
        y.append([ii] * np.shape(_X)[0])

    X = np.concatenate(X, axis=0)
    y = np.concatenate(y)

    np.save('video_color_proc_64.npy', X)
    np.save('video_idx.npy', y)

if __name__ == "__main__":
    main('./')
