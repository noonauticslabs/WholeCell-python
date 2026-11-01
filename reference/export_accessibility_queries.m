function export_accessibility_queries(c, filename)
fields={'monomerBoundSites','complexBoundSites','abasicSites'};
original=cell(size(fields));
for i=1:numel(fields), original{i}=c.(fields{i}); end
cleanup=onCleanup(@() restore(c,fields,original));
mi=find(c.monomerDNAFootprints>0,1); ci=find(c.complexDNAFootprints>0,1);
L=c.sequenceLen;
c.monomerBoundSites=edu.stanford.covert.util.CircularSparseMat([500 1; L-1 2],[mi;mi],[L 4],1);
c.complexBoundSites=edu.stanford.covert.util.CircularSparseMat([900 2; 520 3],[ci;ci],[L 4],1);
c.abasicSites=edu.stanford.covert.util.CircularSparseMat([100 1; 120 2; 1 4],true(3,1),[L 4],1);
c.invalidate();
inputState=struct();
for i=1:numel(fields)
    [s,v]=find(c.(fields{i})); inputState.(fields{i})=struct('subs',arr(s),'vals',arr(v));
end
p=[95 1;105 1;95 2;125 2;490 1;510 1;490 2;890 2;910 2;510 3;2 4;L-2 2];
l=[40;-40;40;-40;40;-40;40;40;-40;40;-10;10];
cases={}; configs=[false false;true false;false true];
for mode=1:3
    for extent=[false true]
        [tf,idx,pos,len,m,cx]=c.isRegionProteinFree(p,l,extent,0,0,configs(mode,1),configs(mode,2));
        cases{end+1}=struct('method','isRegionProteinFree','extent',extent,'both',configs(mode,1),'any',configs(mode,2),...
            'flags',arr(tf),'indices',arr(idx),'positions',arr(pos),'lengths',arr(len),'monomers',arr(m),'complexs',arr(cx));
        [tf,idx,pos,len]=c.isRegionUndamaged(p,l,configs(mode,1),[],extent,configs(mode,2));
        cases{end+1}=struct('method','isRegionUndamaged','extent',extent,'both',configs(mode,1),'any',configs(mode,2),...
            'flags',arr(tf),'indices',arr(idx),'positions',arr(pos),'lengths',arr(len));
    end
end
record=struct('inputState',inputState,'positions',arr(p),'lengths',arr(l),'cases',{cases});
fid=fopen(filename,'w');fwrite(fid,jsonencode(record));fclose(fid);
fprintf('ACCESSIBILITY_EXPORTED %d cases\n',numel(cases));
end
function record=arr(v)
record=struct('shape',size(v),'data',full(v(:)'));
end
function restore(c,fields,values)
for i=1:numel(fields),c.(fields{i})=values{i};end
c.invalidate();
end
