import cv2 as cv
import numpy as np
import numpy.ma as ma
import scipy.interpolate
from scipy.interpolate import Akima1DInterpolator



def findTransform(fn):

    pattern_size = (15,8)
    trg_img_size = (1920, 1080)
    scale_fact = 10     # to speed up calculations

    print('processing %s... ' % fn)
    orig = cv.imread(fn, 0)
    img = cv.resize(orig, (orig.shape[1]/scale_fact,orig.shape[0]/scale_fact) )
    img = 255-img  # invert
    if img is None:
        print("Failed to load", fn)
        return None

    src_img_size = (orig.shape[1], orig.shape[0])

    print "size: %d x %d ... " % (img.shape[1], img.shape[0])
    found, corners = cv.findChessboardCorners(img, pattern_size )
    #print corners

    #cv.imshow('image', img)
    #cv.waitKey(0)

    if found:
        term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
        cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        #vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        #cv.drawChessboardCorners(vis, pattern_size, corners, found)
        #cv.imshow('image', vis)
        #cv.waitKey(0)

        # todo: find the corners with subpixel accuracy in hires image
        corners *= scale_fact


        # generate target coordinates
        pnts_dst = np.array([[trg_img_size[0]*float(r+1)/(pattern_size[0]+1),
                trg_img_size[1]*float(c+1)/(pattern_size[1]+1)] for c in reversed(range(pattern_size[1]))
                                                                for r in reversed(range(pattern_size[0]))])

        # calculate the transform from the perspective image to the flat image we want
        h, status = cv.findHomography(corners, pnts_dst)


        #im_out = cv.warpPerspective(orig, h, (trg_img_size[0],trg_img_size[1]))
        #cv.imshow('image', im_out)
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return h

    else:
        print('chessboard not found')
        return None


"""
imgpath = "Q:\\Projects\\scripts\\python\\sparkmaker\\img\\DSC_6236_conv.JPG"
M = findTransform(imgpath)
img = cv.imread(imgpath, 1)
im_out = cv.warpPerspective(img, M, (1920,1080))
cv.imshow('image', im_out)
cv.waitKey(0)
cv.destroyAllWindows()
"""

def gather_illum_profile():

    pairs = [#["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6281_conv.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6282_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6283_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6284_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6285_conv.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6286_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6287_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6288_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6289_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6290_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6291_conv.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6292_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6293_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6294_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6295_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6296_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6297_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6298_conv.JPG"],
             ["Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6299_convCC.JPG", "Q:/Projects/scripts/python/sparkmaker/img/angles/DSC_6300_conv.JPG"]
             ]

    accu = np.zeros((1080, 1920), dtype="float")
    for chk,img in pairs:

        print chk, img
        M = findTransform(chk)
        im = cv.imread(img, 0)
        im_out = cv.warpPerspective(im, M, (1920,1080))
        accu += im_out

    accu /= accu.max()
    accu2 = (accu*255).astype("uint8")
    cv.imshow('image', accu2)
    cv.waitKey(0)
    #cv.destroyAllWindows()

    return accu


def extract_lcd_responce():
    illums = [  ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6267_conv.JPG", 0, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6268_conv.JPG", 10, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6269_conv.JPG", 20, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6270_conv.JPG", 30, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6271_conv.JPG", 40, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6272_conv.JPG", 50, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6273_conv.JPG", 60, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6274_conv.JPG", 70, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6275_conv.JPG", 80, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6276_conv.JPG", 90, 0],
                ["Q:/Projects/scripts/python/sparkmaker/img/DSC_6277_conv.JPG", 100, 0],
                ]


    base = cv.imread(illums[0][0], 0)
    base = cv.resize(base, (base.shape[1]/5,base.shape[0]/5))
    base_cropped = base[150:-120, 100:-100]

    for il in illums[1:]:

        print il[0]
        img = cv.imread(il[0], 0)
        img = cv.resize(img, (img.shape[1] / 5, img.shape[0] / 5))
        cropped = img[150:-120, 100:-100]

        img = cropped.astype("int16") - base_cropped
        il[2] = img.max()

    #cv.imshow('image', img_cropped)
    #cv.waitKey(0)
    #cv.destroyAllWindows()

    xx = [i[1]/100.0 for i in illums]
    yy = [float(i[2])/illums[-1][2] for i in illums]

    #nx = [c/50.0 for c in range(51)]
    #cs = Akima1DInterpolator(xx,yy)
    #inter = cs(nx)

    # inter = np.interp(nx, xx, yy)
    import matplotlib.pyplot as plt
    # plt.plot(nx,inter)
    plt.plot( xx, yy )
    plt.xlabel('intensity')
    plt.ylabel('measured')
    plt.show()

    return (xx,yy)




# get base ilumination pattern
illum = gather_illum_profile()
# get lcd response curve
lcd_profile = extract_lcd_responce()

thresh = 0.4

out_img = (illum * 255).astype("uint8")

# create a lut to apply an inverse mapping to the illumination
lut = np.array([255 if (x/255.0)<thresh else int(np.interp(thresh/(x/255.0), lcd_profile[1], lcd_profile[0])*255) for x in range(256)], dtype="uint8")
# apply the lut the pattern
inv_img = lut[out_img]
# flip image since we captured it through a mirror
inv_img = cv.flip(inv_img, 0)

cv.imshow('image', inv_img)
cv.waitKey(0)
cv.destroyAllWindows()
#cv.imwrite("Q:\\Projects\\scripts\\python\\sparkmaker\\img\\lumamap.png", inv_img)