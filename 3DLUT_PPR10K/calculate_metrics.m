function [] = calculate_metrics(source_dir, target_dir, mask_dir)

img_paths = dir(fullfile(source_dir, '*.png'));
img_names = {img_paths.name};

group = cell(325, 1);
group_id = 0;
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
mean_ab = [];
parfor i = 1:numel(img_names)
    fprintf('processing image %d/%d\n',i,numel(img_names));
    src_img = imread(fullfile(source_dir,img_names{i}));
    tar_img = imread(fullfile(target_dir,[img_names{i}(1:end-3) 'tif']));
    mask = imread(fullfile(mask_dir,[img_names{i}(1:end-3) 'png']));
    if size(mask,1) ~= size(src_img,1) || size(mask,2) ~= size(src_img,2)
        mask = imresize(mask,[size(src_img,1),size(src_img,2)]);
    end
    mask = repmat(mask,[1,1,3]);
    weights = ones(size(src_img));
    weights(mask==0) = 0.5;
    
    src_lab = rgb2lab(src_img);
    tar_lab = rgb2lab(tar_img);
    
    mean_ab = [mean_ab,squeeze(mean(mean(src_lab,1),2))];
    
    psnr_sum = psnr_sum + psnr(src_img,tar_img);
    Eab_sum = Eab_sum + mean(mean(sqrt(sum((src_lab - tar_lab).^2,3))));
    
    psnr_hc_sum = psnr_hc_sum + psnr(im2double(src_img) .* weights,im2double(tar_img) .* weights);
    Eab_hc_sum = Eab_hc_sum + mean(mean(sqrt(sum((src_lab .* weights - tar_lab .* weights).^2,3))));
    
end
psnr_avg = psnr_sum / numel(img_names);
Eab_avg = Eab_sum / numel(img_names);
psnr_hc_avg = psnr_hc_sum / numel(img_names);
Eab_hc_avg = Eab_hc_sum / numel(img_names);

GLC = 0;
for i = 1:numel(group)
    GLC = GLC + mean(var(mean_ab(2:3,group{i}),[],2));
end
GLC = GLC / numel(group);
fprintf(source_dir)
fprintf(': psnr: %.4f, Eab: %.4f, psnr_hc: %.4f, Eab_hc: %.4f, GLC:%.4f\n', psnr_avg, Eab_avg, psnr_hc_avg, Eab_hc_avg, GLC);

end