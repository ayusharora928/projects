import numpy as np
import cv2
import matplotlib.pyplot as plt
def get_ksize(sigma):
    return int(((sigma - 0.8)/0.15) + 2.0)
def get_gaussian_blur(image, ksize=0, sigma=5):
    if ksize == 0:
        ksize = get_ksize(sigma)
    sep_k = cv2.getGaussianKernel(ksize, sigma)
    return cv2.filter2D(image, -1, np.outer(sep_k, sep_k))
def ssr(image, sigma):
    image = np.float64(image)
    blurred = np.float64(get_gaussian_blur(image, ksize=0, sigma=sigma))
    return np.log10(image + 1e-8) - np.log10(blurred + 1e-8)
def msr(image, sigma_scales=[15, 80, 250],apply_normalization=True):
    msr = np.zeros(image.shape)
    for sigma in sigma_scales:
        msr += ssr(image, sigma)
    msr = msr / len(sigma_scales)
    if apply_normalization:
        msr = cv2.normalize(msr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    return msr
def color_balance(image, low_per, high_per):
    tot_pix = image.shape[1] * image.shape[0]
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100
    ch_list = []
    if len(image.shape) == 2:
        ch_list = [image]
    else:
        ch_list = cv2.split(image)
    cs_image = []
    for i in range(len(ch_list)):
        ch = ch_list[i]
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)))
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if (li == hi):
            cs_image.append(ch)
            continue
        lut = np.array([0 if i < li 
                        else (255 if i > hi else round((i - li) / (hi - li) * 255)) 
                        for i in np.arange(0, 256)], dtype = 'uint8')
        cs_ch = cv2.LUT(ch, lut)
        cs_image.append(cs_ch)    
    if len(cs_image) == 1:
        return np.squeeze(cs_image)
    elif len(cs_image) > 1:
        return cv2.merge(cs_image)
    return None
def msrcr(image, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
    image = image.astype(np.float64) + 1.0
    msr_image = msr(image, sigma_scales, apply_normalization=False)
    crf = beta * (np.log10(alpha * image) - np.log10(np.sum(image, axis=2, keepdims=True)))
    msrcr = G * (msr_image*crf - b)
    msrcr = cv2.normalize(msrcr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
    msrcr = color_balance(msrcr, low_per, high_per)
    return msrcr
def msrcp(image, sigma_scales=[15, 80, 250], low_per=1, high_per=1):
    int_image = (np.sum(image, axis=2) / image.shape[2]) + 1.0
    msr_int = msr(int_image, sigma_scales)
    msr_cb = color_balance(msr_int, low_per, high_per)
    B = 256.0 / (np.max(image, axis=2) + 1.0)
    BB = np.array([B, msr_cb/int_image])
    A = np.min(BB, axis=0)
    msrcp = np.clip(np.expand_dims(A, 2) * image, 0.0, 255.0)   
    return msrcp.astype(np.uint8)

image = cv2.imread(r"C:\Users\piyus\OneDrive\Desktop\New folder\image process\input_image.png")
cv2.imshow("image original", image)
msrcrimage=msrcr(image)
cv2.imshow("msrcrimage",msrcrimage)
msrcpimage = msrcp(image)
half = cv2.resize(msrcpimage, (0, 0), fx = 0.1, fy = 0.1)
bigger = cv2.resize(msrcpimage, (1050, 1610))
stretch_near = cv2.resize(msrcpimage, (780, 540), interpolation = cv2.INTER_LINEAR)
cv2.imshow("msrcpimage", msrcpimage)

Titles =["Original", "Half", "Bigger", "Interpolation Nearest"]
images =[msrcpimage, half, bigger, stretch_near]
count = 4
 
for i in range(count):
    plt.subplot(2, 2, i + 1)
    plt.title(Titles[i])
    plt.imshow(images[i])
 
plt.show()
cv2.imshow("Imageout", image)
cv2.waitKey(0)
cv2.destroyAllWindows()



