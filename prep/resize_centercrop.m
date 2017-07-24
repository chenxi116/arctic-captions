input_dir = '/media/Work_HD/cxliu/datasets/coco/images/train2014/';
output_dir = '/media/Work_HD/cxliu/datasets/coco/images/train2014-center/';
if ~exist(output_dir, 'dir')
    mkdir(output_dir)
end

imgs = dir([input_dir, '*.jpg']);
for i = 1:length(imgs)
    fprintf(1, 'processing %d/%d\n', i, length(imgs));
    im = imread([input_dir, imgs(i).name]);
    
    % resize
    sz = size(im);
    if sz(1) > sz(2)
        im2 = imresize(im, [NaN 256]);
    else
        im2 = imresize(im, [256 NaN]);
    end
    
    % center crop
    crop = 224;
    sz2 = size(im2);
    pos1 = round(sz2(1)/2);
    pos2 = round(sz2(2)/2);
    im_crop = im2(pos1-crop/2:pos1+crop/2-1, pos2-crop/2:pos2+crop/2-1, :);
    
%     figure(1);
%     imshow(im);
%     figure(2);
%     imshow(im_crop);
    
    imwrite(im_crop, [output_dir, imgs(i).name], 'jpg');
end
