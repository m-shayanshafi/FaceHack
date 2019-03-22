import os
from shutil import copyfile

# Hardcoded constants

actorNames = ["Milla Jovovich", "Clive Owen", "Rachel Ray", "Zac Efron", "Julia Roberts", "Julia Stiles", "Nicole Richie"]

targetDirectory = "train_data"

directoryToOrganize = "./dev"

def getFileNames(dataFile, actorName):

	fileNames = []

	file_handle = open(dataFile)

	for line in file_handle:

		if line.startswith('#'): 
			continue

		(name, num, url, coords, md5) = line.strip().split('\t')
		
		if name == actorName:
			fileNames.append(md5+".jpg")

	return fileNames


def copyToTarget(actorDirectory,fileNames):
	
	for fileName in fileNames:
		
		src = directoryToOrganize + "/" + fileName
		dst = actorDirectory + "/" + fileName
		copyfile(src, dst)


def organizeByName(directory):  


	for actorName in actorNames:
		
		actorDirectory = targetDirectory+"/"+actorName

		if not os.path.exists(actorDirectory):
			os.mkdir(actorDirectory)

		fileNames = getFileNames(directoryToOrganize+"_"+"urls.txt", actorName)
		copyToTarget(actorDirectory, fileNames)


def main():
	
	organizeByName(directoryToOrganize)

if __name__ == '__main__':

	if not os.path.exists(targetDirectory):
		os.mkdir(targetDirectory)
	
	main()