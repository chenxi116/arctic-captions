function f30k_generate_attn(setname, sc)
if nargin == 0
    setname = 'train'; % 'train' 'dev' 'test'
    sc = 1.0;
end

img_dir = '../external/flickr30k-center/';

entities_dir = '../external/Flickr30kEntities/';
addpath(entities_dir);

f_img_list = ['../data/f30k/f30k_', setname, '.txt'];
img_list = importdata(f_img_list);

attn_gt_idx = cell(length(img_list)*5, 1);
attn_gt_map = zeros(length(img_list)*5*10, 196, 'single');
map_idx = 1;

vis = 0;
G = fspecial('gaussian', [100 100], 50); % Gaussian filter
attnmap_final = zeros(224, 224, 3, 'uint8');
coef = 9;

for i = 1:length(img_list) % for each image
    img_id = img_list{i}(1:end-4);
    fprintf(1, 'processing image %d/%d: %s\n', i, length(img_list), img_id);
    
    s_data = getSentenceData([entities_dir, 'Sentences/', img_id, '.txt']);
    a_data = getAnnotations([entities_dir, 'Annotations/', img_id, '.xml']);
    
    for j = 1:5 % for each of the five captions
        s = s_data(j);
        attn_gt_idx{(i-1)*5 + j} = zeros(1, length(strsplit(s.sentence)));
        for l = 1:length(s.phraseID)% for each of the noun phrase in the caption
            % find id index
            ididx = 0;
            for k = 1:length(a_data.id)
                if str2double(a_data.id{k}) == str2double(s.phraseID{l})
                    ididx = k;
                    break;
                end
            end
            % assert(ididx > 0);
            if ididx == 0 
                continue;
            end
            % find label index
            labelidx = a_data.idToLabel(ididx);
            labelidx = labelidx{1};
            bbox_orig = zeros(length(labelidx), 4);
            bbox_all = zeros(length(labelidx), 4);
            for k = 1:length(labelidx) % for each label, extract bbox
                bbox = a_data.labels(labelidx(k)).boxes;
                if isempty(bbox)
                    continue;
                end
                bbox_orig(k, :) = bbox;
                bbox_all(k, :) = adjust_bbox(bbox, a_data.dims(1:2), sc);
            end
            if max(bbox_all(:)) < 1
                fprintf(1, 'sentence %d phrase %d no bbox\n', j, l);
                continue;
            end
            % plot_bbox(img_list{i}, bbox_orig, bbox_all);pause;
            
            % generate mask
            mask = zeros(224);
            for k = 1:length(labelidx)
                mask(bbox_all(k, 2):bbox_all(k, 4), bbox_all(k, 1):bbox_all(k, 3)) = 1;
            end
            assert(nnz(mask) > 0);
            % downsample mask
            mask14 = single(imresize(mask, [14 14], 'bilinear'));
            mask14 = mask14/(sum(mask14(:)));
            assert(min(mask14(:)) >= 0);
            
            % visualize (optional)
            if vis == 1
                attnmap_up = double(imresize(mask14, [224 224])); % up sample
                attnmap_smooth = conv2(attnmap_up, G, 'same'); % apply Gaussian filter
                max_val = max(attnmap_smooth(:));
                for k = 1:3
                    attnmap_final(:, :, k) = uint8((coef-1)/coef*255*attnmap_smooth/max_val);
                end
                img = imread([img_dir, img_list{i}]);
                imshow(img/coef + attnmap_final);
                pause;
            end
            
            % save
            for k = 1:length(s.phrases{l})
                attn_gt_idx{(i-1)*5 + j}(s.phraseFirstWordIdx(l)+k-1) = map_idx;
            end
            attn_gt_map(map_idx, :) = reshape(mask14', [1, 196]);
            map_idx = map_idx + 1;
        end
    end
end

attn_gt_map = attn_gt_map(1:map_idx-1, :);

save(['../data/f30k/f30k_attn_gt_', num2str(sc*10), '.', setname, '.mat'], ...
    'attn_gt_idx', 'attn_gt_map');
end

function plot_bbox(imname, bbox_orig, bbox_all)
f30k_dir = '/media/Work_HD/cxliu/datasets/';
im1 = imread([f30k_dir, '/flickr30k-images/', imname]);
im2 = imread([f30k_dir, '/flickr30k-center/', imname]);
figure(1);

for i = 1:length(bbox_orig(:, 1))
    bbox = bbox_orig(i, :); % extract a row
    bbox(3) = bbox(3) - bbox(1);
    bbox(4) = bbox(4) - bbox(2);
    im1 = insertShape(im1, 'Rectangle', bbox, 'Color', 'red');
end
subplot(1, 2, 1);
imshow(im1);

for i = 1:length(bbox_all(:, 1))
    bbox = bbox_all(i, :);
    bbox(3) = bbox(3) - bbox(1);
    bbox(4) = bbox(4) - bbox(2);
    im2 = insertShape(im2, 'Rectangle', bbox, 'Color', 'red');
end
subplot(1, 2, 2);
imshow(im2);
end