from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import cv2


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def get_mask(frame, bodypix_url='http://localhost:9000'):
    _, data = cv2.imencode(".jpg", frame)
    r = requests.post(
        url=bodypix_url,
        data=data.tobytes(),
        headers={'Content-Type': 'application/octet-stream'})
    mask = np.frombuffer(r.content, dtype=np.uint8)
    mask = mask.reshape((frame.shape[0], frame.shape[1]))
    return mask


def post_process_mask(mask):
    mask = cv2.erode(mask, np.ones((5,5),np.uint8), iterations=1)
    #mask = cv2.blur(mask.astype(float), (15, 15))
    return mask


def get_frame(cap, background_scaled):
    mask = None
    while mask is None:
        try:
            mask = get_mask(cap)
        except requests.RequestException:
            print("mask request failed, retrying")
    mask = post_process_mask(mask)

    inv_mask = 1 - mask
    cap = adjust_gamma(cap, 1.5)
    for c in range(cap.shape[2]):
        cap[:, :, c] = cap[:, :, c] * mask + background_scaled[:, :, c] * inv_mask
    return cap


def main():
    seg = cv2.VideoCapture(0)
    background = cv2.imread("background5.jpeg")
    while True:
        ret, original_im = seg.read()
        background_sc = cv2.resize(background, (original_im.shape[1], original_im.shape[0]))
        ## original_im = cv2.cvtColor(original_im, cv2.COLOR_BGR2RGB)
        msk = get_frame(original_im, background_sc)
        mask = cv2.flip(msk, 1)
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        #mask = warmImage(warmImage)
        #plt.imshow(mask)
        #plt.show()
        cv2.imshow("final", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__=='__main__':
    main()
