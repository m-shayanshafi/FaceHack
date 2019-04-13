% Detect landmarks and align faces to a normalized pose (via an affine
% transformation)

py_cmd = 'python'; % update me
py_script = './face_landmark_detection.py'; % update me
dlib_model = './shape_predictor_68_face_landmarks.dat'; % update me
ref_landmarks = './ref_marks.csv';

% subjects = {'Aaron_Eckhart', 'Brad_Pitt', 'Clive_Owen', 'Drew_Barrymore', 'Julia_Roberts', 'Julia_Stiles', 'Milla_Jovovich', 'Nicole_Richie', 'Rachael_Ray', 'Zac_Efron'};

 %subjects = {'Brad_Pitt'};

for i = 1:numel(subjects)
	disp(subjects{i})
    ims_path = fullfile('./train_data/', subjects{i});
    %face_landmark_detection(py_cmd, py_script, dlib_model, ...
    %                ims_path, '*.jpg');
    align_vgg_pose(ims_path, '*.jpg', ref_landmarks);
end

exit
	