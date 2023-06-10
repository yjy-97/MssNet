function [total_num_entropy_cal, slice_list] = cal_entropy_slices(file_list_name)
    % close all; clear all; clc;
    
    total_num_entropy_cal = 0;
    slice_path_file = strcat('\', file_list_name, '_path.txt');
    file_path = strcat('F:\S_NC\gray_matter\',  slice_path_file);
    disp(file_path)
    fpn = fopen(file_path,'rt');
    num_dir = 0;
    file_list = {};
    while feof(fpn) ~= 1
        num_dir = num_dir + 1;
        tline = fgetl(fpn);
        % disp(tline);
        file_list{num_dir, 1} = tline;
    end
    fclose(fpn);
    
    entropy_value_file = strcat('entropy_value_', file_list_name, '.txt');
    
    %% entropy
    for i = 1:num_dir
        cur_num_entropy_cal = 0;
        dir_path = file_list{i ,1};
        %% save file
        save_path = strcat(dir_path, '\', entropy_value_file);

        if exist(save_path)>0
            delete(save_path);
            disp(fprintf('Detele file [%s] .',save_path));
        end
        save_file = fopen(save_path, 'a');
        slice_list = dir(fullfile(dir_path));
        num_slice = size(slice_list,1);
        sorted_entropy_value_arr = zeros(num_slice - 2, 1);
        sorted_name_cell = {};
        for slice_list_index = 3:num_slice
            % slice_list(3) = 'entropy_value_AD_gray_matter_Slices.txt'
            image_name = slice_list(slice_list_index).name;
            % disp(image_name)
            slice_path = strcat(dir_path, '\', image_name);
            %disp(slice_path);
            try
                cur_num_entropy_cal = cur_num_entropy_cal + 1;
                image = imread(slice_path);
                %disp(image);
                %image=rgb2gray(image);
                [C,L]=size(image); %Find the size of the image
                Img_size=C*L; %The total number of image pixels
                G=256; %Grayscale of the image
                H_x=0;
                nk=zeros(G,1);%Generate an all-zero matrix with G rows and 1 column
                for i=1:C
                for j=1:L
                Img_level=image(i,j)+1; %Get the grayscale of the image
                nk(Img_level)=nk(Img_level)+1; %Count the number of points per grayscale pixel
                end
                end
                for k=1:G  
                Ps(k)=nk(k)/Img_size; %Calculate the probability of each pixel
                if Ps(k)~=0; %If the probability of the pixel is not zero
                H_x=-Ps(k)*log2(Ps(k))+H_x; %The formula for entropy
                end
                end
                entropy_value = H_x;
                %disp(entropy_value);
                try
                    % entropy_val_struct.value(cur_num_entropy_cal) = entropy_value;
                    % entropy_val_struct.name{cur_num_entropy_cal, 1} = image_name;
                    sorted_entropy_value_arr(cur_num_entropy_cal) = entropy_value;
                    sorted_name_cell{cur_num_entropy_cal, 1} = image_name;
                catch
                    disp('entropy_val_struct error.');
                    % exit(0);
                end
            catch
                disp(fprintf('File [%s] is not a .jpg file.',slice_path));
            end
        end
        
        % sorted_entropy_value：
        % sorted_name_index：
        [sorted_entropy_value, sorted_name_index] = sort([sorted_entropy_value_arr], 'descend');
        
        % sorted_name_index
        for index = 1:cur_num_entropy_cal
            try
                value_ = sorted_entropy_value(index);
                name_ = sorted_name_cell{sorted_name_index(index), 1};
                fprintf(save_file,'%s, %d \r\n', name_, value_);
                % fprintf(save_file, '\r\n');
            catch
                disp(fprintf('[error] index = %d', index));
            end
            
        end
        
        fclose(save_file);
        
        % disp(sprintf('num_entropy_cal = %d', cur_num_entropy_cal));
        total_num_entropy_cal = total_num_entropy_cal + cur_num_entropy_cal;
    end
    %%
    disp(sprintf('num_entropy_cal = %d', total_num_entropy_cal));
    
    %% input
    % [total_num_entropy_cal, slice_list] = cal_entropy_slices('AD_Slices')
    % [total_num_entropy_cal, slice_list] = cal_entropy_slices('NC_gray_matter_Slices')
end