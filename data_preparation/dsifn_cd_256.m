clear all;
close all;
clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Train-A
imgs_name = struct2cell(dir('DSIFN/download/Archive/train/t1/*.jpg'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/train/t1/', img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/train/A/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%Train-B
imgs_name = struct2cell(dir('DSIFN/download/Archive/train/t2/*.jpg'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/train/t2/', img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/train/B/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%Train-label
imgs_name = struct2cell(dir('DSIFN/download/Archive/train/mask/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/train/mask/',img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/train/label/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%test-A
imgs_name = struct2cell(dir('DSIFN/download/Archive/test/t1/*.jpg'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/test/t1/', img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/test/A/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%test-B
imgs_name = struct2cell(dir('DSIFN/download/Archive/test/t2/*.jpg'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/test/t2/', img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/test/B/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%test-label
imgs_name = struct2cell(dir('DSIFN/download/Archive/test/mask/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/test/mask/',img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/test/label/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%val-A
imgs_name = struct2cell(dir('DSIFN/download/Archive/val/t1/*.jpg'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/val/t1/', img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/val/A/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%val-B
imgs_name = struct2cell(dir('DSIFN/download/Archive/val/t2/*.jpg'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/val/t2/', img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/val/B/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end

%val-label
imgs_name = struct2cell(dir('DSIFN/download/Archive/val/mask/*.png'));
for i=1:1:length(imgs_name)
    img_file_name = imgs_name{1,i};
    temp = imread(strcat('DSIFN/download/Archive/val/mask/',img_file_name));
    c=1;
    for j=1:2
        for k=1:2
            patch = temp((j-1)*256+1: j*256, (k-1)*256+1: k*256, :);
            imwrite(patch, strcat('DSIFN_256/val/label/', img_file_name(1:end-4), '_', num2str(c), '.png'));
            c=c+1;
        end
    end
    
end
