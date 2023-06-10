function [deleted_slice_num] = delete_slice_N(file_list_name, num_not_delete)
    deleted_slice_num = 0;
    slice_path_file = strcat('\', file_list_name, '_path.txt');
    file_path = strcat('H:\oasis2_work\NC1sub3\',  slice_path_file);
    fpn = fopen(file_path,'rt');
    disp(file_path);
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
    
    for i = 1:num_dir
        dir_path = file_list{i ,1};
        Slices_path = strcat(dir_path, '\', entropy_value_file);
        if exist(Slices_path)>0
            disp(fprintf('File [%s] exist...', Slices_path));
            Slices_path_file = fopen(Slices_path, 'rt');
            slice_cell = {};
            slice_num = 0;
            while feof(Slices_path_file) ~= 1
                slice_num = slice_num + 1;
                tline = fgetl(Slices_path_file);
                % disp(tline);
                slice_cell{slice_num, 1} = tline;
            end
            fclose(Slices_path_file);

            if (num_not_delete < slice_num)
                for i = (num_not_delete+1):slice_num
                    % delete those low entropy slice
                    slice_line = slice_cell{i, 1};
                    slice_line_split = regexp(slice_line, ',', 'split');
                    slice_name = strtrim(char(slice_line_split(1)));
                    slice_entropy_value = strtrim(char(slice_line_split(2)));
                    try
                        if strcmp(slice_name, '')
                            disp(fprintf('slice_name = %s is null. \r\n', slice_name));
                        else
                            delete_slice_path = strcat(dir_path, '\', slice_name);
                            if exist(delete_slice_path)>0
                                delete(delete_slice_path);
                                deleted_slice_num = deleted_slice_num + 1;
                                disp(fprintf('[deleted] %s \r\n', slice_line));
                            else
                                disp(fprintf('[Not exist] slice_line = %s not exist. \r\n', slice_line));
                            end
                        end
                        
                    catch
                        disp(fprintf('[detele error] slice_name = %d', slice_name));
                    end
                    % delete end ...
                end
            else
                disp('[error] num_not_delete >= slice_num ......');
            end
 
        end
        
    end
    
    % [deleted_name_list] = delete_slice_N('AD_gray_matter_Slices', 101)
    % [deleted_name_list] = delete_slice_N('NC_gray_matter_Slices', 101)
 
end
