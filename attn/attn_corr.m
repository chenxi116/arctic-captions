function attn_corr(setname, modelname)

% setname = 'test';
% modelname = 'f30k_03_k5';

entities_dir = '../external/Flickr30kEntities/';
addpath(entities_dir);
img_dir = '../external/flickr30k-center/';
output_dir = ['../vis/', modelname, '_bbox/'];
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

load(['../cap/f30k/', modelname, '.alphas.', setname, '.mat']); % alphas
img_list = importdata(['../data/f30k/f30k_', setname, '.txt']);
G = fspecial('gaussian', [100 100], 50); % Gaussian filter for visualization
attnmap_vis = zeros(224, 224, 3, 'uint8'); % for visualization
coef = 9; % constant for visualization

% load alignment txt file
f_align = ['../cap/f30k/', modelname, '.align.', setname, '.txt'];
fid = fopen(f_align, 'r');
num_align = size(importdata(f_align), 1);
max_score = zeros(num_align, 1);
baseline = zeros(num_align, 1);

for i = 1:num_align % for every alignment
    align = textscan(fgetl(fid), '%d %s %d %d %d'); 
    % image index, image name, phrase starting position in new
    % sentence, phrase length, coreference index
    
    a_data = getAnnotations([entities_dir, 'Annotations/', align{2}{1}, '.xml']);

    % find id index
    ididx = 0;
    for k = 1:length(a_data.id)
        if str2double(a_data.id{k}) == align{5}
            ididx = k;
            break;
        end
    end
    assert(ididx > 0);
    % find label index
    labelidx = a_data.idToLabel(ididx);
    labelidx = labelidx{1};
    bbox_all = zeros(length(labelidx), 4);
    for k = 1:length(labelidx) % for each label, extract bbox
        bbox = a_data.labels(labelidx(k)).boxes;
        if isempty(bbox)
            continue;
        end
        bbox_all(k, :) = adjust_bbox(bbox, a_data.dims(1:2));
    end
    if min(bbox_all(:)) < 1
        fprintf(1, 'alignment %d/%d no bbox\n', i, num_align);
        continue;
    end
    % for insertShape function
    bbox_pos = bbox_all;
    bbox_pos(:, 3) = bbox_all(:, 3) - bbox_all(:, 1);
    bbox_pos(:, 4) = bbox_all(:, 4) - bbox_all(:, 2);
    
    mask = zeros(224);
    for k = 1:length(labelidx)
        mask(bbox_all(k, 2):bbox_all(k, 4), bbox_all(k, 1):bbox_all(k, 3)) = 1;
    end
    % imagesc(mask);
    
    alpha = alphas{align{1}};
    score = zeros(align{4}, 1);
    img = imread([img_dir, img_list{i}]);
    for j = 1:align{4} % for every word in the phrase
        attnmap = reshape(alpha(align{3} + j - 1, :), [14 14])';
        attnmap_up = double(imresize(attnmap, [224 224])); % upsample
        attnmap_final = attnmap_up/sum(attnmap_up(:)); % normalize
        attnmap_smooth = conv2(attnmap_up, G, 'same'); % apply Gaussian filter
        % attnmap_final = attnmap_smooth/sum(attnmap_smooth(:)); % normalize
        max_val = max(attnmap_smooth(:));
        for k = 1:3
            attnmap_vis(:, :, k) = uint8((coef-1)/coef*255*attnmap_smooth/max_val);
        end

        score(j) = sum(sum(mask.*attnmap_final));
        assert(score(j) <= 1.01 && score(j) >= -0.01);

        img_attn = img/coef + attnmap_vis;
        img_bbox = insertShape(img_attn, 'Rectangle', bbox_pos, ...
            'Color', 'red', 'Linewidth', 2);
        imshow(img_bbox);
        imwrite(img_bbox, [output_dir, align{2}{1}, '_', num2str(align{3} + j - 1, '%02d'), '.png'], 'png');
    end
    max_score(i) = max(score); % attention correctness score
    baseline(i) = sum(mask(:))/(224*224); % score if attend uniformly
    assert(baseline(i) <= 1 && baseline(i) >= 0);

    fprintf(1, 'alignment %d/%d done\n', i, num_align);
end
fclose(fid);

save(['../vis/', modelname, '.mat'], 'max_score', 'baseline');

attn_corr = max_score(max_score > 0 & max_score < 1);
attn_corr_baseline = baseline(baseline > 0 & baseline < 1);
fprintf(1, 'Attention Correctness (%d alignments): %f\n', length(attn_corr), mean(attn_corr));
fprintf(1, 'Attention Correctness (baseline, %d alignments): %f\n', ...
    length(attn_corr_baseline), mean(attn_corr_baseline));
end
