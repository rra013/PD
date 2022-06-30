
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

annotatedFilename = "Annotated Cracks"

videoFilename = "D:\\2022 Summer Startup Work\\Pavment Distress One File\\Video\\testVid.mov"

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


testImageFolderName = "Test Images"
saveImageFolderName = "Processed Images"
finalImageFolderName = "Cracks"


def getImageList():
    imgList = os.listdir(testImageFolderName)
    for i in range(len(imgList)):
        imgList[i] = testImageFolderName+'\\'+imgList[i]
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

imagesToAverage = getImageList()

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

def whoAmI():
    print("I am the color finder")

<<<<<<< HEAD
def readFramesFromImage(imagePath):
    cam = cv2.VideoCapture(imagePath)
  
    try:
        
        # creating a folder named data
        if not os.path.exists('data'):
            os.makedirs('data')
    
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
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()


def testNoTemplate():
    horizontalOverVerticalThresh = 0.5
    
=======
def __main__():
    horizontalOverVerticalThresh = 0.5
    start_time = time.time()
>>>>>>> 7d445c174c8b57c4a3ddf535cf42c9992ab5fd8d
    imagesToTest = getImageList()
    points = []
    count = 0
    for image in imagesToTest: 
        print("Image number:", count)
        processed = cracksInImage(image, .5, True, 1)
        point = []
        point.append(maxFinder(processed, 10, True, findNextRight, str(count)+"TestHorizontal"))
        point.append(maxFinder(processed, 10, True, findNextDown, str(count)+"TestVertical"))
        points.append(point)
        count += 1
    print(points)
    processedPoints = []
    for point in points:
        totalCrack = sum(point)
        processedPoints.append(np.array(point)/totalCrack)
    print(processedPoints)
    for i in range(len(imagesToTest)):
        point = processedPoints[i]
        status = ""
        if point[0]/point[1] < 1-horizontalOverVerticalThresh:
            status = "Vertical Crack"
        elif point[0]/point[1] > 1+horizontalOverVerticalThresh:
            status = "Horizontal Crack"
        else:
            status = "Pothole"
        print(imagesToTest[i], "Damage:", status)
<<<<<<< HEAD

def __main__():
    start_time = time.time()
    readFramesFromImage(videoFilename)
=======
>>>>>>> 7d445c174c8b57c4a3ddf535cf42c9992ab5fd8d
    print("--- %s seconds ---" % (time.time() - start_time))
if(__name__ == '__main__'):
    __main__()

