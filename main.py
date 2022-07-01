
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
ImageFile.LOAD_TRUNCATED_IMAGES = True

annotatedFilename = "Annotated Cracks"
roadMarkingFolder = "Road Markings"
videoFilename = "D:\\2022 Summer Startup Work\\Pavment Distress One File\\Video\\testVid.mov"
testImageFolderName = "Test Images"
saveImageFolderName = "Processed Images"
finalImageFolderName = "Cracks"

def getImageList():
    imgList = os.listdir(testImageFolderName)
    for i in range(len(imgList)):
        imgList[i] = testImageFolderName+'\\'+imgList[i]
    return imgList

imagesToAverage = getImageList()

def findNextRight(image, pixel, maxI, maxJ):
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

def verticalCrackCheck(image, trimParameters=[0,0], threshold=0.5, save=False, name="crack"):
    conformityToVerticalCrack = []
    cracks = []
    trimmed = image[:]
    trimmed = trimmed[:,trimParameters[1]:len(trimmed[0])-trimParameters[1]]
    trimmed = trimmed[trimParameters[0]:len(trimmed)-trimParameters[0]]
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
    conformityToHorizontalCrack = []
    cracks = []
    trimmed = image[:]
    trimmed = trimmed[:,trimParameters[1]:len(trimmed[0])-trimParameters[1]]
    trimmed = trimmed[trimParameters[0]:len(trimmed)-trimParameters[0]]
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
    imgList = os.listdir(folderName)
    for i in range(len(imgList)):
        imgList[i] = folderName+'\\'+imgList[i]
    return imgList

def getAdjacent(i, j, maxI, maxJ):
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
    return np.reshape(np.zeros(len(image)*len(image[0])*3), (len(image), len(image[0]), 3))


#DO NOT USE: DEPRECATED
def averageSingleImage(imageName):
    image = io.imread(imageName)
    maxI = len(image)
    maxJ = len(image[0])
    imageAveraged = emptyImageCopy(image)
    for i in range(maxI):
        for j in range(maxJ):
            adjacents = getAdjacent(i, j, maxI, maxJ)
            colorSum = np.zeros(3)
            for cell in adjacents:
                colorSum += image[cell[0], cell[1]]
            colorAverage = colorSum//9
            imageAveraged[i][j] = colorAverage
    io.imsave(saveImageFolderName+"\\"+imageName, imageAveraged)
    return imageAveraged

def processImage(imageName, save):
    image = io.imread(imageName)
    #print(image)
    grayImage = rgb2gray(image)
    closedImage = closing(grayImage)
    
    if save:
        io.imsave(saveImageFolderName+"\\"+imageName[len(testImageFolderName):], closedImage)
    return closedImage

def cracksInImage(imageName, threshold, save, closes=1):
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



def readFramesFromImage(imagePath, folderName, numFrames=-1):
    cam = cv2.VideoCapture(imagePath)
  
    try:
        
        # creating a folder named data
        if not os.path.exists(folderName):
            os.makedirs(folderName)
    
    # if not created then raise error
    except OSError:
        print ('Error: Creating directory of data')
    
    # frame
    currentframe = 0
    
    while(True):
        
        # reading from frame
        ret,frame = cam.read()
    
        if ret:
            # if video is still left continue creating images
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
            #print(currentframe, numFrames, currentframe==numFrames)
            if currentframe == numFrames:
                return "Done"
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()

def __main__():
    start_time = time.time()
    frameFolderName = "data"
    numFrames = 400
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
    print("--- %s seconds ---" % (time.time() - start_time))

if(__name__ == '__main__'):
    __main__()

