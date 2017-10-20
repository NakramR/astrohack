import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.collections
import seaborn as sns

dataFolder = ''

def read_image(idd):
    id = str(idd)
    
    if id[-4:] == '.npy':
        X = np.load(dataFolder+id)
    elif os.path.isfile(dataFolder+id+'.npy'):
        X = np.load(dataFolder+id + '.npy')
    elif os.path.isfile(dataFolder+id+'-g.csv'):
        X = np.genfromtxt(dataFolder+id+'-g.csv', delimiter=",")
    else:
        X = None

    X = np.float32(X)
    return X

def drawOneGalaxy(galaxyID):
    oneImageData = read_image(galaxyID)
    print(galaxyID)

    # new image
    fig = plt.figure(figsize=(15,15))
    #set grid spec for the 4 graphs
    gs = gridspec.GridSpec(2, 2, width_ratios=[5, 1], height_ratios=[1,5]) 

    #draw image
    plt.title("raw")
    plt.subplot(gs[2])
    plt.imshow(oneImageData)
    
    # horizontal (top) sum
    plt.subplot(gs[0])
    plt.title(galaxyID)
    plt.plot(oneImageData.sum(axis=0))

    # vertical (bottom-right) sum
    ax = plt.subplot(gs[3])
    ss = np.flip(oneImageData.sum(axis=1),axis=0)
    plt.scatter(x=ss, y=list(range(oneImageData.shape[1])), s=1)
    lines = [[(ss[i-1],i-1),(ss[i],i)] for i in range(1,len(ss))]
    lc = matplotlib.collections.LineCollection(lines)
    ax.add_collection(lc)
    
    #value histogram
    plt.subplot(gs[1])
    plt.hist(oneImageData.reshape(-1), bins=100)
    plt.yscale('log')    
    sns.despine()
    plt.tight_layout()

    #add small log of image
    ax = fig.add_axes([0.02,0.6,.2,.2])
    plt.imshow(np.log(oneImageData-oneImageData.min()+0.00001))
    plt.show()

# for _ in range(5):
#     i = random.randint(0,len(dataFileList))
#     oneImageData = np.load(dataFolder+'1237648704067273096.npy')
#     drawOneGalaxy(dataFileList[i])

BlackHighThreshold = 0.1 # the definition of 'black' to determine the radius of a non galaxy object, in luminance
PeakExclusionRadius = 0.1 # the radius around the center within which we ignore luminosity peaks, expressed as an image percentage
NonClearingRadius = 0.1 # the radius around the center within which we don't clean (even with peaks outside)
ObjectLuminosityPercentage = 0.7 # the peak luminosity threshold that we remove points around, expressed as a percentage of the peak center luminosity
valueWhenRemovingPixel = 0 # -55 when debugging, 0 when running

def removeNegs(A):
    for i in range(len(A)):
        for j in range(len(A)):
            A[i][j] = max(0,A[i][j])

    return A

def normalizeInt(A):
    ma = np.amax(A)
    mi = np.amin(A)

    A = A * (ma - mi)
    return A

def removePeakAtPosition(data, x, y, size):
    global BlackHighThreshold
    global valueWhenRemovingPixel
    imagewidth = len(data)
    center = imagewidth/2

    threshold = BlackHighThreshold #definition of "black"
    exlusionRadiusSquared = (imagewidth*PeakExclusionRadius)**2  # radius around the center where we don't remove anything
    for i in range(imagewidth):
        if (
                ( x-i >= 0 and data[x-i][y] < threshold) or
                ( x+i < imagewidth and data[x+i][y] < threshold ) or
                ( y -i >= 0 and data[x][y-i] < threshold ) or
                ( y +i < imagewidth and data[x][y+i] < threshold )
            ):
            circlesize = i
            break

    for i in range(x-circlesize, x+circlesize+1):
        for j in range(y-circlesize, y+circlesize+1):
            if ( i >= 0 and i < len(data[1]) and
                 i >= 0 and j < len(data[1]) and
                     (x-i)**2 + (y-j)**2 <= circlesize**2):
                # exlusion zone
                if ( (center-i)**2 + (center-j)**2 >= exlusionRadiusSquared ):
                    data[i][j] = valueWhenRemovingPixel

    return data

def removeAboveThreshold(A, threshold):
    global valueWhenRemovingPixel
    for i in range(len(A)):
        for j in range(len(A)):
            if ( A[i][j] > threshold ):
                A[i][j] = valueWhenRemovingPixel

    return A

def findLumCenter(A):
    center = int(len(A)/2)

    peakfound = False
    moved = False
    px = center
    py = center
    while True:
        moved = False
        for i in range(-5, 5):
            for j in range(-5, 5):
                if (A[px + i][py + j] > A[px][py]):
                    px = px + i
                    py = py + j
                    moved = True
                    break
        if (moved == False):
            peakfound = True
            break

    if (peakfound == True):
        centerLum = A[px][py]

    return centerLum

def findMaxima(A):
    global NonClearingRadius
    global ObjectLuminosityPercentage
    width = len(A)
    center = int(width / 2)

    #find lum center
    centerLum = findLumCenter(A)

    starLumThreshold = centerLum * ObjectLuminosityPercentage
    exlusionRadiusSquared = (width*NonClearingRadius)**2


    for x in range(width):
        for y in range(width):
            # find the peak:
            if A[x][y] <= starLumThreshold:
                continue
            # found a place where there's a peak. Find it
            peakfound = False
            moved = False
            px = x
            py = y
            while True:
                moved = False
                for i in range(-2,2):
                    for j in range(-2,2):
                        if ( px+i < 1 or py+j < 1 or px+i >= width or py+j >= width):
                            continue

                        if (A[px + i][py + j] > A[px][py]):
                            px = px + i
                            py = py + j
                            moved = True
                            break
                if (moved == False):
                    peakfound = True
                    break

            if (peakfound == True):
                # remove that peak
                if ((center - px) ** 2 + (center - py) ** 2 > exlusionRadiusSquared):
                    A = removePeakAtPosition(A,px,py,max(abs(py-y),abs(px-x))+1)
#            print("found", (px, py), (x, y))

    return A

def cleanupImage(A):
    A = removeNegs(A)
    A = findMaxima(A)
    A = removeAboveThreshold(A, findLumCenter(A))
    return A
    
def xi2(true,pred,error):
    s=np.mean((true-pred)**2/error**2)
    return s

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def scale_image(A):
    A = (A - A.min()) / (A.max()-A.min())
    return A

def normalize_image(A):
    A -= np.mean(A)
    A /= np.std(A)
    
    return A

def crop_image(Xg, percentage=0.75):
    h,w = Xg.shape
    cy, cx = h//2, w//2
    dy, dx = int(cy*percentage), int(cx*percentage) # crop a bit around center
    Xg = Xg[cy-dy:cy+dy,cx-dx:cx+dx]
    
    return Xg

def img_preprocnoread(Xg, preProcNum = 0):
    if preProcNum & 128: #hackaton preproc
        Xg = cleanupImage(Xg)

    if preProcNum & 1: # vflip
        Xg = np.flip(Xg,0)
    if preProcNum & 2: # hflip
        Xg = np.flip(Xg,1)
    if preProcNum & 4: # rotate
        Xg = np.rot90(Xg)
    if preProcNum & 8: # scale [0,1]
        Xg = scale_image(Xg)
    if preProcNum & 16: # log
        Xg = np.log1p(Xg - Xg.min())
    if preProcNum & 32: # normalize
        Xg = normalize_image(Xg)
    if preProcNum & 64: # crop
        Xg = crop_image(Xg)

    
    if Xg.shape[0] >= 224:
        Xgr = cv2.resize(Xg,(224,224), cv2.INTER_AREA)
    else:
        Xgr = cv2.resize(Xg,(224,224), cv2.INTER_CUBIC)
    
    return Xgr
    
def img_preproc(id, preProcNum = 0):
    Xg = read_image(id)
    return img_preprocnoread(Xg,preProcNum)

def getAstrohackDataFrame():

    df =  pd.read_fwf('metaData.dat', comment = '#')

    df['RA'] = df['RA'].apply(np.float64)
    df['DEC'] = df['DEC'].apply(np.float64)
    df['D25'] = df['D25'].apply(np.float64)
    df['redshi'] = df['redshi'].apply(np.float64)
    df['logMstar'] = df['logMst'].apply(np.float64) #renamed
    df['err_logMstar'] = df['err_l'].apply(np.float64) #renamed
    df['GalSize_kpc'] = df['GalSize_kpc'].apply(np.float64)
    df['Distance'] = df['D_Mpc'].apply(np.float64) #renamed
    df['d_pix_kpc'] = df['d_pix_kpc'].apply(np.float64)
    df['ML_g'] = df['ML_g'].apply(np.float64)

    df['lin_mass'] = np.power(10, df.logMstar)
    df['lin_err'] = df['lin_mass'] * np.log(10) * df.err_logMstar

    df['hasFile'] = df.SDSS_ID.apply(lambda x: os.path.isfile(dataFolder+str(x)+'.npy'))

    df = df.drop(['logMst','err_l'], axis=1)
    
    df = df[df.logMstar != -99]
    df = df[df.hasFile == True]
    df = df[df['lin_err']!=0]
    df = df[df.Distance < 600]
    df = df[df.ML_g < 15]

    df = df[~df['SDSS_ID'].isin(['1237668349209149549','1237662224593846425','1237648705129283884','1237648703514804436'])] # remove 2 buggy galaxies

    df['ML_g_rel_err'] = df['ML_g_rel_err'].apply(np.float64)

    df = df[df['ML_g_rel_err'] != 0]
    
    return df

def lgb_chi2(Yp, train_data):
    Y = train_data.get_label()
    err = 1/train_data.get_weight()**0.5
    return 'Chi²', round(xi2(Y,Yp,err),1), False

def getLGBMModelsWithCV(trainSet, YSet, errSet, errlinSet):
    kf = KFold(n_splits=nSplits,shuffle=True, random_state=220477)

    cvtrainpreds = np.zeros([len(Xg3f),nSplits])
    models = []
    counter = 0
    for tix, vix in kf.split(trainSet):
        X_train, X_test = trainSet[tix], trainSet[vix]
        Y_train, Y_test = YSet[tix], YSet[vix]

        lgb_train = lgbm.Dataset(X_train, Y_train)
        lgb_eval = lgbm.Dataset(X_test, Y_test)

        lgb_train.set_weight(1/train_err**2)
        lgb_eval.set_weight(1/test_err**2)
        
        gbm = lgbm.train(lgbm_params,
                       lgb_train,
                       num_boost_round=maxBoostRuns,
                       valid_sets=[lgb_train,lgb_eval],  # eval training data
                       verbose_eval=100,
                       early_stopping_rounds=100,
                        feval = lgb_chi2
                        )
        models.append(gbm)

        p = gbm.predict(X_test)

        cvtrainpreds[vix,counter] = p
        counter = counter+1
        
    return models, cvtrainpreds

def getLGBMModelsNoCV(trainSet, YSet, errSet, errlinSet):
    cvtrainpreds = np.zeros([len(trainSet),1])
    models = []
    counter = 0

    tix, vix = list(range(0,int(len(trainSet)*0.9))), list(range(int(len(trainSet)*0.9),len(trainSet)))
    
    X_train, X_test = trainSet[tix], trainSet[vix]
    Y_train, Y_test = YSet[tix], YSet[vix]

    lgb_train = lgbm.Dataset(X_train, Y_train)
    lgb_eval = lgbm.Dataset(X_test, Y_test)

    lgb_train.set_weight(1/errSet[tix]**2)
    lgb_eval.set_weight(1/errSet[vix]**2)

    gbm = lgbm.train(lgbm_params,
                       lgb_train,
                       num_boost_round=maxBoostRuns,
                       valid_sets=[lgb_train,lgb_eval],  # eval training data
                       verbose_eval=100,
                       early_stopping_rounds=100,
                        feval = lgb_chi2
                     
                    )
    models.append(gbm)

    p = gbm.predict(X_test)
    cvtrainpreds[vix,counter] = p
    counter = counter+1
        
    return models, cvtrainpreds


