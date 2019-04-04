function band=readBlockImage(root,folderdirfromname,productname,filename,postfix,xlabels,ylabels,nrow,ncol,default)
    band=zeros([nrow,ncol],'single');
    xdim=length(xlabels);
    ydim=length(ylabels);
    cursor_x=1;
    cursor_y=1;
    disp(['Block Image Name: ',filename]);
    for j=1:ydim
        for i=1:xdim
            disp(['Reading Block(',num2str(j),',',num2str(i),')...']);
            ylabel=ylabels{j};
            xlabel=xlabels{i};
            filedirfrom=[root,'/',folderdirfromname,'/',productname,'/',filename,'-',ylabel,'-',xlabel,postfix];
            bandblock=single(imread(filedirfrom));
            [blocknrow,blockncol,~]=size(bandblock);
            bandblock(isnan(bandblock))=default;
            band(cursor_y:cursor_y+blocknrow-1,cursor_x:cursor_x+blockncol-1)=bandblock;
            if i==xdim
                cursor_x=1;
                cursor_y=cursor_y+blocknrow;
            else
                cursor_x=cursor_x+blockncol;
            end
            
        end
    end
end