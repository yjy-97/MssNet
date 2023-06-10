function num_list = get_slices_root(filelistname, save_path, slice_stride
 
    fpn = fopen(filelistname,'rt');
    sbj_conter = 0;
    filelist = {};
    while feof(fpn) ~= 1
        sbj_conter = sbj_conter + 1;
        tline = fgetl(fpn);
        % disp(tline);
        filelist{sbj_conter, 1} = tline;
    end
 
    fclose(fpn);
 
    %%
    for i = 1 : sbj_conter
        %% 5-fold
        if i <= sbj_conter * 0.2
            sub = 1;
        else
            if i <= sbj_conter * 0.4
                sub = 2;
            else
                if i <= sbj_conter * 0.6
                    sub = 3;
                else
                    if i <= sbj_conter * 0.8
                        sub = 4;
                    else 
                        sub = 5;
                    end
                end
            end
        end
        subfold = strcat('sub',num2str(sub),'\');
        %%           
        %slice_conter = 0;
        file = filelist{i ,1};
        % file = strcat('..\', file, '\');
        disp(sprintf('[%d]...... %s', i, file));
        try
            nii = load_untouch_nii(file);
        catch
            disp('load_nii() failure...Using load_untouch_nii()')
            nii = load_untouch_nii(file);
        end
        
        %mkdir(save_path,strcat('S',num2str(sbj_conter,'%03d')))
        sbj_fold = strcat(save_path, subfold, file(9:18), '\');   
        disp(nii)
        img = nii.img;
        % img = mapmm(img);
        [x,y,z] = size(img);
        % Z
        mkdir(sbj_fold,'ZSlice');
        ZSlicepath = strcat(sbj_fold,'ZSlice','\');
        for j = 1:slice_stride:z
            slice = img(:,:,j);
            %disp(slice);
            slice_path = strcat(ZSlicepath,'slice_Z',num2str(j),'.jpg');
            if exist(slice_path)>0
                disp(fprintf('[exist - delete] slice_path = %s  \r\n', slice_path));
                delete(slice_path);
            end
            slice=mat2gray(slice);
            imwrite(slice, slice_path, 'Quality', 100)
        end
        %% Y
        mkdir(sbj_fold,'YSlice');
        YSlicepath = strcat(sbj_fold,'YSlice','\');
        for j = 1:slice_stride:y
            slice = reshape(img(:,j,:),[x,z]);
            slice_path = strcat(YSlicepath,'slice_Y',num2str(j),'.jpg');
            if exist(slice_path)>0
                disp(fprintf('[exist - delete] slice_path = %s  \r\n', slice_path));
                delete(slice_path);
            end
            slice=mat2gray(slice);
            imwrite(slice, slice_path, 'Quality', 100)
        end    
        %% X
        mkdir(sbj_fold,'XSlice');
        XSlicepath = strcat(sbj_fold,'XSlice','\');
        for j = 1:slice_stride:x
            slice = reshape(img(j,:,:),[y,z]);
            slice_path = strcat(XSlicepath,'slice_X',num2str(j),'.jpg');
            if exist(slice_path)>0
                disp(fprintf('[exist - delete] slice_path = %s  \r\n', slice_path));
                delete(slice_path);
            end
            slice=mat2gray(slice);
            imwrite(slice, slice_path, 'Quality', 100)
        end
    end
    
    num_list = sbj_conter;

end


% filelistname:E:\testmri\NC_CSF.txt
%'G:\otest\AD\AD_original.txt'
% save_path: E:\testmri\save
%get_slices_root('F:\NC\NC_gray_matter.txt','F:\AD\gray_matter',1)
