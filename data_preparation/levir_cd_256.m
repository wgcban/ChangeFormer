clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train-A
imgs_name = struct2cell(dir('LEVIR-CD/train/A/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/train/A/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/train/A/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%Train-B
imgs_name = struct2cell(dir('LEVIR-CD/train/B/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/train/B/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/train/B/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end

%Train-label
imgs_name = struct2cell(dir('LEVIR-CD/train/label/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/train/label/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/train/label/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Test-A
imgs_name = struct2cell(dir('LEVIR-CD/test/A/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/test/A/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/test/A/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end

%Test-B
imgs_name = struct2cell(dir('LEVIR-CD/test/B/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/test/B/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/test/B/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end

%Test-label
imgs_name = struct2cell(dir('LEVIR-CD/test/label/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/test/label/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/test/label/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%val-A
imgs_name = struct2cell(dir('LEVIR-CD/val/A/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/val/A/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/val/A/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end

%val-B
imgs_name = struct2cell(dir('LEVIR-CD/val/B/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/val/B/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/val/B/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end

%val-label
imgs_name = struct2cell(dir('LEVIR-CD/val/label/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('LEVIR-CD/val/label/',img_file_name));
    c=1;
    for j=1:4
        for k=1:4
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('LEVIR-CD256/val/label/', img_file_name(1:end-4), '_', num2str(j+k-1), '.png'));
            c=c+1;
        end
    end
    
end