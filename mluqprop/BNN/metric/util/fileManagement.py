import os

def getClusterList(folder):
    fileList =  os.listdir(folder)
    clusterList = []
    for file in fileList:
        if file.startswith('frechetDist'):
            clusterNum = int(file[15:-4])
            clusterList.append(clusterNum)

    clusterList.sort()
    return clusterList



if __name__ == "__main__":
    getClusterList('../data')
