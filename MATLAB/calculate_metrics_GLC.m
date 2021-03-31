function [GLC_Lab, GLC_RGB, GLC_Lab_Lab, GLC_Lab_ab, GLC_RGB_RGB] = calculate_metrics_GLC(source_dir)

% source_dir = '/home/liangjie/hdd4t/AIPS_data/GLC_10group_100times_devided/80';

img_paths = dir(fullfile(source_dir,'*.tif'));
img_names = {img_paths.name};

group = cell(51,1);
group_id = 1;
last_img_id = 0;
for i = 1:numel(img_names)
    a = split(img_names{i},'_');
    cur_img_id = str2double(a{1});   
    if cur_img_id ~= last_img_id
        group_id = group_id + 1;
        last_img_id = cur_img_id;
    end
    group{group_id} = [group{group_id},i]; 
end

psnr_sum = 0;
Eab_sum = 0;
psnr_hc_sum = 0;
Eab_hc_sum = 0;
mean_Lab = [];
mean_RGB = [];
for i = 1:numel(img_names)
%     fprintf('processing image %d/%d\n',i,numel(img_names));
    src_img = imread(fullfile(source_dir,img_names{i}));
    
    src_lab = rgb2lab(src_img);
    
    mean_Lab = [mean_Lab,squeeze(mean(mean(src_lab,1),2))];
    mean_RGB = [mean_RGB,squeeze(mean(mean(src_img,1),2))];
    
end

GLC_Lab = [0,0,0];
GLC_RGB = [0,0,0];
GLC_Lab_Lab = 0;
GLC_Lab_ab = 0;
GLC_RGB_RGB = 0;
for i = 2:numel(group)
    for j = 1:3
        GLC_Lab(j) = GLC_Lab(j) + mean(var(mean_Lab(j,group{i}),[],2));
        GLC_RGB(j) = GLC_RGB(j) + mean(var(mean_RGB(j,group{i}),[],2));
    end
    GLC_Lab_Lab = GLC_Lab_Lab + mean(var(mean_Lab(1:3,group{i}),[],2));
    GLC_Lab_ab = GLC_Lab_ab + mean(var(mean_Lab(2:3,group{i}),[],2));
    GLC_RGB_RGB = GLC_RGB_RGB + mean(var(mean_RGB(1:3,group{i}),[],2));
end
GLC_Lab = GLC_Lab / numel(group)
GLC_RGB = GLC_RGB / numel(group)
GLC_Lab_Lab = GLC_Lab_Lab / numel(group)
GLC_Lab_ab = GLC_Lab_ab / numel(group)
GLC_RGB_RGB = GLC_RGB_RGB / numel(group)
fprintf(source_dir)
fprintf(' GLC:%.4f\n', GLC_Lab);

end