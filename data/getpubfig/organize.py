import os
from shutil import copyfile

# Hardcoded constants

# For dev  
actorNames = ["Alyssa Milano", "Zac Efron", "Julia Roberts", "Nicole Richie", "Christina Ricci", "Clive Owen", "Cristiano Ronaldo"]
directoryToOrganize = "./dev"

# For eval
# actorNames = ["Aaron Eckhart", "Brad Pitt", "Drew Barrymore"]
# directoryToOrganize = "./eval"
targetDirectory = "train_data"


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
		
		if os.path.isfile(src):
			dst = actorDirectory + "/" + fileName
			copyfile(src, dst)	

		


def organizeByName(directory):  


	for actorName in actorNames:
		
		# actorSplitName = actorName.split(" ")

		actorFolderName = actorName.replace(' ', '_')
		actorDirectory = targetDirectory+"/"+actorFolderName

		if not os.path.exists(actorDirectory):
			os.mkdir(actorDirectory)

		fileNames = getFileNames(directoryToOrganize+"_"+"urls_complete.txt", actorName)
		copyToTarget(actorDirectory, fileNames)


def main():
	
	organizeByName(directoryToOrganize)

if __name__ == '__main__':

	if not os.path.exists(targetDirectory):
		os.mkdir(targetDirectory)
	
	main()