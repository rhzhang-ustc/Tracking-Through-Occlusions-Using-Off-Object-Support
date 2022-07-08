import cv2
import numpy as np


def gif_generation(frame_lst, output_path, colored):

    frame_shape = frame_lst[0].shape
    video_size = (frame_shape[1], frame_shape[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, video_size, colored)

    for frame in frame_lst:
        video.write(frame)

    video.release()


def sample_demo(sample, S, N, output_path):
    imgs = np.array(sample['rgbs'].reshape(-1, 3, 368, 496))
    frame_lst = []

    for i in range(S):
        frame = imgs[i]
        frame = frame.swapaxes(0, 2)
        frame = frame.swapaxes(0, 1)
        frame = np.ascontiguousarray(frame)

        trajs = np.squeeze(np.array(sample['trajs']), axis=0)

        for j in range(N):
            pt = trajs[i, j, :]
            x, y = int(pt[0]), int(pt[1])
            cv2.rectangle(frame, (x - 3, y - 3), (x + 3, y + 3), (0, 255, 0))

        frame_lst.append(frame)

    gif_generation(frame_lst, output_path, True)
