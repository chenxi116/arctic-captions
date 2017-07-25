function bbox_new = adjust_bbox(bbox, dims, sc)
if nargin < 3
    sc = 1.0;
end
bbox_new = zeros(1, 4);
assert(bbox(1) <= dims(2));
assert(bbox(3) <= dims(2));
assert(bbox(2) <= dims(1));
assert(bbox(4) <= dims(1));
r = min(dims/256);
bbox = bbox/r;
dims = dims/r;

c1 = (bbox(1) + bbox(3))/2;
c2 = (bbox(2) + bbox(4))/2;
bbox(1) = c1 + (bbox(1) - c1) * sc;
bbox(2) = c2 + (bbox(2) - c2) * sc;
bbox(3) = c1 + (bbox(3) - c1) * sc;
bbox(4) = c2 + (bbox(4) - c2) * sc;

bbox_new(1) = round(min(max(1, bbox(1) - (dims(2)/2 - 112)), 224));
bbox_new(2) = round(min(max(1, bbox(2) - (dims(1)/2 - 112)), 224));
bbox_new(3) = round(min(max(1, bbox(3) - (dims(2)/2 - 112)), 224));
bbox_new(4) = round(min(max(1, bbox(4) - (dims(1)/2 - 112)), 224));
assert(bbox_new(1) <= bbox_new(3));
assert(bbox_new(2) <= bbox_new(4));
end