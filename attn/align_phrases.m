function align_phrases(setname, modelname)
% find exact match between new sentence and ground truth sentences
% store coreference chain index in .txt file
% just change 'modelname' and run!

% setname = 'test';
% modelname = 'f30k_03_k5';

entities_dir = '../external/Flickr30kEntities/';
addpath(entities_dir);
f_img_list = ['../data/f30k/f30k_', setname, '.txt'];
img_list = importdata(f_img_list);

load(['../cap/f30k/', modelname, '.NP.', setname, '.mat']); % NP
f_caps = ['../cap/f30k/', modelname, '.', setname, '.txt'];
caps_list = importdata(f_caps);

fid = fopen(['../cap/f30k/', modelname, '.align.test.txt'], 'w+');

for i = 1:length(img_list) % for each image
    img_id = img_list{i}(1:end-4);
    fprintf(1, 'processing image %d/%d: %s\n', i, length(img_list), img_id);
    
    s_data = getSentenceData([entities_dir, 'Sentences/', img_id, '.txt']);
    pid = []; % phraseID concatenation
    p = []; % phrases concatenation
    for j = 1:5 % for each of the five captions
        s = s_data(j);
        pid = [pid, s.phraseID];
        p = [p, s.phrases];
    end
    
    id = cell(length(NP{i}), 1);
    for j = 1:length(NP{i}) % all the phrases in the new sentence
        id{j} = '0';
        for k = 1:length(p) % all the phrases in the ground truth sentences
            if isequal(lower(strjoin(p{k})), lower(strjoin(NP{i}{j})))
                id{j} = pid{k};
                break;
            end
        end
    end
    
    for j = 1:length(NP{i})
        if str2double(id{j}) > 0
            fprintf(1, 'Phrase in new sentence: %s\n', strjoin(NP{i}{j}));
            
            sentence = textscan(caps_list{i}, '%s');
            idx = FirstWordIdx(sentence{1}, NP{i}{j});
            
            % image index, image name, phrase starting position in new
            % sentence, phrase length, coreference index
            fprintf(fid, '%d %s %d %d %d\n', i, img_id, idx, ...
                length(NP{i}{j}), str2double(id{j}));   
        end
    end
end

fclose(fid);
end

function idx = FirstWordIdx(sentence, phrase)
len_s = length(sentence);
len_p = length(phrase);
for idx = 1:len_s
    flag = 1;
    for j = 1:len_p
        if ~isequal(sentence{idx + j - 1}, phrase{j})
            flag = 0;
            break;
        end
    end
    if flag == 1
        break;
    end
end
end