#!/bin/bash

actorNames=('Aaron_Eckhart' 'Brad_Pitt' 'Clive_Owen' 'Drew_Barrymore' 'Julia_Roberts' 'Nicole_Richie' 'Zac_Efron' 'Alyssa_Milano' 'Christina_Ricci' 'Cristiano_Ronaldo')

# need one more female/male actor

for actor in ${actorNames[@]}; do
  subjectLine="subjects = {'$actor'}"

  python ./face_landmark_detection.py ./shape_predictor_68_face_landmarks.dat ./train_data/$actor "*.jpg"

  echo $subjectLine

  matlab -nodisplay -nodesktop -r "$subjectLine; run preprocess_data.m"

done

processedDir="processed_train_data" 

if [ ! -d "$processedDir" ]; then
	mkdir $processedDir
fi

cd $processedDir

for actor in ${actorNames[@]}; do

	actorDir=$actor

	if [ ! -d "$actorDir" ]; then
		mkdir $actorDir
	fi

	#copy all files with aligned keyword train to processed
	cp ../train_data/$actor/aligned* ./$actor	

done