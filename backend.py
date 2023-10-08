
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
import os
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.morphology import closing
from skimage.color import rgb2gray
import matplotlib as plt
import cv2
from PIL import ImageFile
from random import randrange
import sys
from flask import Flask, render_template, request, jsonify
print(sys.getrecursionlimit())

ImageFile.LOAD_TRUNCATED_IMAGES = True


#Folder where annotated images go
annotatedFilename = "Annotated Cracks"
#Folder where image highlighting road markings are saved
roadMarkingFolder = "Road Markings"
#Location of the video processed
videoFilename = "D:\\2022 Summer Startup Work\\Pavment Distress One File\\Video\\testVid.mov"
#Location of images processed
testImageFolderName = "Test Images"
#Where greyscaled and closed images are saved
saveImageFolderName = "Processed Images"
#Where black/white images are saved
finalImageFolderName = "Cracks"
#Where frames from the video containing damaged pavement are stored
outputFolder = "Damaged Asphault"
#Global list of damaged pixels
allDamagedPixels = []


def getImageList():
    """Get all images in the folder of tested images"""
    imgList = os.listdir(testImageFolderName)
    for i in range(len(imgList)):
        imgList[i] = testImageFolderName+'\\'+imgList[i]
    return imgList


def findNextRight(image, pixel, maxI, maxJ):
    """Find the next black pixel to the right of PIXEL in IMAGE with height MAXI and length MAXJ"""
    if pixel[1] >= maxJ - 1:
        return "End image"
    elif image[pixel[0]][pixel[1]+1] == 0:
        return [pixel[0], pixel[1]+1]
    elif pixel[0] == 0 or pixel[0] >= maxI - 1:
        return "End image"
    elif image[pixel[0]-1][pixel[1]+1] == 0:
        return [pixel[0]-1, pixel[1]+1]
    elif image[pixel[0] + 1][pixel[1]+1] == 0:
        return [pixel[0]+1, pixel[1]+1]
    elif image[pixel[0]+1][pixel[1]] == 0:
        return [pixel[0]+1, pixel[1]]
    else:
        return "End of crack"

def findNextDown(image, pixel, maxI, maxJ):
    """Find the next black pixel below PIXEL in IMAGE with height MAXI and length MAXJ"""
    if pixel[0] >= maxI -1:
        return "End image"
    elif image[pixel[0]+1][pixel[1]] == 0:
        return [pixel[0]+1, pixel[1]]
    elif pixel[1] >= maxJ - 1 or pixel[1] == 0:
        return "End image"
    elif image[pixel[0]+1][pixel[1]+1] == 0:
        return [pixel[0]+1, pixel[1]+1]
    elif image[pixel[0] + 1][pixel[1]-1] == 0:
        return [pixel[0]+1, pixel[1]-1]
    elif image[pixel[0]][pixel[1]+1] == 0:
        return [pixel[0], pixel[1]+1]
    else:
        return "End of crack"

def maxFinder(image, jumpSize, output, function, name="crack"):
    """Find the longest output of FUNCTION checking every JUMPSIZE pixel in IMAGE, saving with name NAME if OUTPUT"""
    maxI = len(image)
    maxJ = len(image[0])
    maxLen = 0
    maxList = []
    for i in range(0, maxI, jumpSize):
        for j in range(0, maxJ, jumpSize):
            print([i, j])
            if [i, j] in maxList:
                continue
            else:
                if image[i][j] == 1 and [i, j] not in maxList:
                    pixel = [i, j]
                    traversed = [[i, j]]
                    count = 0
                    next = function(image, pixel, maxI, maxJ)
                    while type(next) != str:
                        count += 1
                        traversed.append(next)
                        next = function(image, next, maxI, maxJ)
                    if count > maxLen:
                        maxLen = count
                        maxList = traversed
    if output:
        annotatedImage = []
        for i in range(maxI):
            for j in range(maxJ):
                if image[i][j] == 0:
                    annotatedImage.append([0, 0, 0])
                else:
                    annotatedImage.append([255, 255, 255])
        annotatedImage = np.reshape(annotatedImage, (maxI, maxJ, 3))
        print(len(annotatedImage), len(annotatedImage[0]), maxList)
        for pixel in maxList:
            annotatedImage[pixel[0]][pixel[1]] = [255, 0, 0]
        io.imsave(annotatedFilename+"\\"+name+'.jpg', annotatedImage)
        #print(annotatedImage)
    return maxLen

def trimImage(image, trimParameters):
    """Trim IMAGE by TRIMPARAMETERS"""
    trimmed = image[:]
    trimmed = trimmed[:,trimParameters[1]:len(trimmed[0])-trimParameters[1]]
    trimmed = trimmed[trimParameters[0]:len(trimmed)-trimParameters[0]]
    return trimmed

def verticalCrackCheck(image, trimParameters=[0,0], threshold=0.5, save=False, name="crack"):
    """Find a vertical crack in IMAGE after being trimmed by TRIMPARAMETERS; if more than THRESHOLD pixels in a column are black, identify a crack and save under name NAME if SAVE"""
    conformityToVerticalCrack = []
    cracks = []
    trimmed = trimImage(image, trimParameters)
    maxI = len(trimmed)
    maxJ = len(trimmed[0])

    print(len(image), len(image[0]), len(trimmed))
    for i in range(len(trimmed[0])):
        conformityToVerticalCrack.append(sum(trimmed[:,i])/len(trimmed[:,i]))
        if conformityToVerticalCrack[i] < threshold:
            cracks.append(i)
    print(cracks)
    if save:
        annotatedImage = []
        for i in range(maxI):
            for j in range(maxJ):
                if trimmed[i][j] == 0:
                    annotatedImage.append([0, 0, 0])
                else:
                    annotatedImage.append([255, 255, 255])
        annotatedImage = np.reshape(annotatedImage, (maxI, maxJ, 3))
        for i in range(maxI):
            for j in range(maxJ):
                if j in cracks:
                    annotatedImage[i][j] = [255, 0, 0]
        io.imsave(annotatedFilename+"\\"+name+".jpg", annotatedImage)
    return [conformityToVerticalCrack, cracks]

def horizontalCrackCheck(image, trimParameters=[0,0], threshold=0.5, save=False, name="crack"):
    """Find a horizontal crack in IMAGE after being trimmed by TRIMPARAMETERS; if more than THRESHOLD pixels in a row are black, identify a crack and save under name NAME if SAVE"""
    conformityToHorizontalCrack = []
    cracks = []
    trimmed = trimImage(image, trimParameters)
    maxI = len(trimmed)
    maxJ = len(trimmed[0])

    print(len(image), len(image[0]), len(trimmed))
    for i in range(len(trimmed)):
        conformityToHorizontalCrack.append(sum(trimmed[i,:])/len(trimmed[i,:]))
        if conformityToHorizontalCrack[i] < threshold:
            cracks.append(i)
    print(cracks)
    if save:
        annotatedImage = []
        for i in range(maxI):
            for j in range(maxJ):
                if trimmed[i][j] == 0:
                    annotatedImage.append([0, 0, 0])
                else:
                    annotatedImage.append([255, 255, 255])
        annotatedImage = np.reshape(annotatedImage, (maxI, maxJ, 3))
        for i in range(maxI):
            for j in range(maxJ):
                if i in cracks:
                    annotatedImage[i][j] = [0, 255, 0]
        io.imsave(annotatedFilename+"\\"+name+".jpg", annotatedImage)
    return [conformityToHorizontalCrack, cracks]

def getVideoFrames(folderName):
    """Get the frames in the FOLDERNAME folder"""
    imgList = os.listdir(folderName)
    for i in range(len(imgList)):
        imgList[i] = folderName+'\\'+imgList[i]
    return imgList

def getAdjacent(i, j, maxI, maxJ):
    """Get all pixels adjacent to the pixel at I, J in an image of size MAXI by MAXJ"""
    topLeft = [i-1, j-1]
    left = [i, j-1]
    bottomLeft = [i+1, j-1]
    bottom = [i+1, j]
    top = [i-1, j]
    topRight = [i-1, j+1]
    right = [i, j+1]
    bottomRight = [i+1, j+1]
    adjacent = [topLeft, left, bottomLeft, top, bottom, topRight, right, bottomRight]
    for i in adjacent:
        for j in range(len(i)):
            if i[j] < 0:
                i[j] = 0
            if j == 0 and i[j] >= maxI:
                i[j] = maxI - 1
            if j == 1 and i[j] >= maxJ:
                i[j] = maxJ - 1
            
    return adjacent

def emptyImageCopy(image):
    """Return an empty image with the same size as IMAGE"""
    return np.reshape(np.zeros(len(image)*len(image[0])*3), (len(image), len(image[0]), 3))

def processImage(imageName, save):
    """Greyscale and close image at IMAGENAME; save if SAVE"""
    image = io.imread(imageName)
    #print(image)
    grayImage = rgb2gray(image)
    closedImage = closing(grayImage)
    if save:
        io.imsave(saveImageFolderName+"\\"+imageName[len(testImageFolderName):], closedImage)
    return closedImage

def cracksInImage(imageName, threshold, save, closes=1):
    """Find cracks in image at IMAGENAME, saving if SAVE and closing CLOSES times; pixel set to white/black if over/under THRESHOLD"""
    processedImage = processImage(imageName, True)
    for i in range(len(processedImage)):
        for j in range(len(processedImage[0])):
            if processedImage[i][j] < threshold:
                processedImage[i][j] = 0
            else:
                processedImage[i][j] = 1
    for i in range(closes):
        processedImage = closing(processedImage)
    if save:
        io.imsave(finalImageFolderName+"\\"+imageName[len(testImageFolderName):], processedImage)
    return processedImage

def markingsInImage(imageName, save, saveFolder=finalImageFolderName):
    """Highlight white/yellow road markings in image at IMAGENAME, saving in SAVEFOLDER if SAVE"""
    processedImage = processImage(imageName, True)
    for i in range(len(processedImage)):
        for j in range(len(processedImage[0])):
            if processedImage[i][j] < 0.05:
                processedImage[i][j] = 0
            else:
                processedImage[i][j] = 1
    if save:
        io.imsave(saveFolder+"\\"+imageName[len(testImageFolderName):], processedImage)
    return processedImage

def noiseRemoval(image):
    """Attempt to smooth noise in IMAGE, do not use; very time and resource intensive, not as good as closing"""
    returned = image[:]
    for i in range(len(image)):
        for j in range(len(image[0])):
            print(i, j)
            count = 0
            for pixel in getAdjacent(i, j, len(image), len(image[0])):
                count += image[pixel[0]][pixel[1]]
            if count/len(getAdjacent(i, j, len(image), len(image[0]))) >= 0.5:
                returned[i][j] = 1
            else:
                returned[i][j] = 0
    return returned

def createAnnotattableImage(maxI, maxJ, image):
    annotatedImage = []
    for i in range(maxI):
        for j in range(maxJ):
            if image[i][j] == 0:
                annotatedImage.append([0, 0, 0])
            else:
                annotatedImage.append([255, 255, 255])
    annotatableImage = np.reshape(annotatedImage, (maxI, maxJ, 3))
    return annotatableImage

def readFramesFromImage(imagePath, folderName, numFrames=-1, frameSkip=1):
    """Read and save every FRAMESKIP frame in NUMFRAMES (or all if not given) frames from video at IMAGEPATH in folder FOLDERNAME"""
    cam = cv2.VideoCapture(imagePath)
  
    try:
        
        # creating a folder named data
        if not os.path.exists(folderName):
            os.makedirs(folderName)
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentFrame = 0
    
    while(cam.isOpened()):
        
        # reading from frame
        ret,frame = cam.read()
    
        
        if numFrames <= currentFrame and numFrames > 0:
            cam.release()
            break
        elif ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentFrame) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentFrame += frameSkip
            #print(currentframe, numFrames, currentframe==numFrames)
            cam.set(cv2.CAP_PROP_POS_FRAMES, currentFrame)
        else:
            cam.release()
            break
    
    # Release all space and windows once done
    cv2.destroyAllWindows()

def partialCrackCheckHor(image, length, save, name="crack"):
    """Check for LENGTH pixel strings of black pixels in IMAGE horizontally, saving under name NAME if SAVE"""
    maxI = len(image)
    maxJ = len(image[0])
    cracks = []
    for i in range(len(image)):
        for j in range(0, len(image[0]), length):
            if i < maxI and j < (maxJ - length) and sum(image[i][j:j+length]) == 0:
                for pixelJVal in range(j, j+length):
                    cracks.append([i, pixelJVal])
    if save:
        annotatedImage = []
        for i in range(maxI):
            for j in range(maxJ):
                if image[i][j] == 0:
                    annotatedImage.append([0, 0, 0])
                else:
                    annotatedImage.append([255, 255, 255])
        annotatedImage = np.reshape(annotatedImage, (maxI, maxJ, 3))
        for pixel in cracks:
            annotatedImage[pixel[0]][pixel[1]] = [200, 200, 50]
        io.imsave(annotatedFilename+"\\"+name+".jpg", annotatedImage)
    return cracks

def partialCrackCheckVer(image, length, save, name="crack"):
    """Check for LENGTH pixel strings of black pixels in IMAGE vertically, saving under name NAME if SAVE"""
    maxI = len(image)
    maxJ = len(image[0])
    cracks = []
    for i in range(0, len(image), length):
        for j in range(len(image[0])):
            if i < (maxI - length) and j < maxJ and sum(image[i:i+length,j]) == 0:
                for pixelIVal in range(i, i+length):
                    cracks.append([pixelIVal, j])
    if save:
        annotatedImage = []
        for i in range(maxI):
            for j in range(maxJ):
                if image[i][j] == 0:
                    annotatedImage.append([0, 0, 0])
                else:
                    annotatedImage.append([255, 255, 255])
        annotatedImage = np.reshape(annotatedImage, (maxI, maxJ, 3))
        for pixel in cracks:
            annotatedImage[pixel[0]][pixel[1]] = [50, 200, 200]
        io.imsave(annotatedFilename+"\\"+name+".jpg", annotatedImage)
    return cracks


def processFullImageDamageFromVideo():
    frameFolderName = "data"
    numFrames = 1
    readFramesFromImage(videoFilename, frameFolderName, numFrames)
    frames = getVideoFrames(frameFolderName)
    processedFrames = []
    crackData = []
    count = 0
    for frame in frames:
        processedFrames.append(cracksInImage(frame, 0.3, True)[:])
    for frame in processedFrames:
        crackData.append(verticalCrackCheck(frame, [400, 800], 0.6, True, "vertical"+str(count)))
        crackData.append(horizontalCrackCheck(frame, [400, 800], 0.6, True, "horizontal"+str(count)))
        count += 1
    print(crackData)

def getGap(values):
    gaps = []
    for value in range(min(values), max(values)):
        if value not in values:
            gaps.append(value)
    return gaps

def distinctDamage(pixelTested, maxI, maxJ):
    global allDamagedPixels
    returned = np.array(pixelTested)
    if pixelTested in allDamagedPixels:
                allDamagedPixels.remove(pixelTested)
    for adjacent in getAdjacent(pixelTested[0], pixelTested[1], maxI, maxJ):
        if adjacent in allDamagedPixels and adjacent != pixelTested:
            #print(len(allDamagedPixels))
            returned = np.append(returned, distinctDamage(adjacent, maxI, maxJ)) 
    return returned

def createRandomColorList(n):
    returned = []
    for i in range(n):
        returned.append([0, 0, randrange(100, 255)])
    return returned

def execute(file, contrastThresh=0.3, frameLimit=-1, skipNum=1, classifierThresh=0.1, progress={}):
    start_time = time.time()
    frameFolderName = "data"
    numFrames = frameLimit
    frameSkip = skipNum
    saveInBetween = True
    partialCrackLength = 10
    whiteThresh = contrastThresh
    damagedFrames = []
    progress['status'] = 'Reading images from video...'
    
    readFramesFromImage(file, frameFolderName, numFrames, frameSkip)
    frames = getVideoFrames(frameFolderName)
    processedFrames = []
    count = 0
    progress['status'] = 'Processing frames...'
    for frame in frames:
        print(count)
        processedFrames.append(cracksInImage(frame, whiteThresh, True)[:])
    progress['status'] = 'Analyzing for damage...'
    for frame in processedFrames:
        frame = trimImage(frame, [450, 800])
        damageInFrame = []
        damageInFrame.append(partialCrackCheckHor(frame, partialCrackLength, saveInBetween, "horizontal"+str(count)))
        damageInFrame.append(partialCrackCheckVer(frame, partialCrackLength, saveInBetween, "vertical"+str(count)))

        for list in damageInFrame:
            for pixel in list:
                allDamagedPixels.append(pixel)
        sys.setrecursionlimit(len(allDamagedPixels)*100)

        annotatable = createAnnotattableImage(len(frame), len(frame[0]), frame)
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]#createRandomColorList(len(cracks))
        #print('dif',damageInFrame)
        dpCount = 0
        for i in range(len(damageInFrame)):
            for pixel in damageInFrame[i]:
                annotatable[pixel[0]][pixel[1]] = colors[0]
                dpCount += 1
        if  classifierThresh <= dpCount/(len(annotatable) * len(annotatable[0])):
            damagedFrames.append("Cracks"+str(count)+".jpg")
            io.imsave(outputFolder+"\\"+"Frame"+str(count)+".jpg", io.imread(frames[count]))
        io.imsave(annotatedFilename+"\\"+"Cracks"+str(count)+".jpg", annotatable)
        count += 1
    #print(crackData)
    
    progress['status'] = "Process completed in %s seconds." % (time.time() - start_time)
    return damagedFrames

def __main__():
    execute(videoFilename, 0.3, 60, 1)

if(__name__ == '__main__'):
    __main__()

