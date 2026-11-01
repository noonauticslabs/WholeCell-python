function export_region_queries(c, filename)
original = c.polymerizedRegions;
cleanup = onCleanup(@() restore(c, original));
cases = {};
L = c.sequenceLen;
positions = [1 1; 1 2; 1 3; 50 2; 95 2; 110 2; 200 2; 210 2; 400 3; 490 3; L-3 1; 2 1; 2 4; L-2 4];
lengths = [10; 120; 20; -60; 20; 150; 60; -30; 30; 30; 10; -10; -10; 10];
methods = {'isRegionPolymerized','isRegionNotPolymerized','isRegionSingleStranded','isRegionDoubleStranded'};
for scenario=1:2
    if scenario==2
        c.polymerizedRegions = edu.stanford.covert.util.CircularSparseMat(...
            [1 1; 1 2; 200 2; 400 3; 1 4; L-5 4], [L;100;50;100;20;6], [L 4], 1);
        c.invalidate();
    end
    [subs, vals] = find(c.polymerizedRegions);
    for im=1:numel(methods)
        method=methods{im};
        for extent=[false true]
            [tf,idx,pos,len]=c.(method)(positions,lengths,extent);
            cases{end+1}=struct('scenario',scenario,'method',method,'extent',extent,...
                'polymerizedSubs',arr(subs),'polymerizedLengths',arr(vals),...
                'positions',arr(positions),'lengths',arr(lengths),...
                'flags',arr(tf),'indices',arr(idx),'resultPositions',arr(pos),'resultLengths',arr(len));
        end
    end
end
fid=fopen(filename,'w'); fwrite(fid,jsonencode(struct('cases',{cases}))); fclose(fid);
fprintf('REGION_QUERIES_EXPORTED %d cases\n',numel(cases));
end
function record=arr(value)
record=struct('shape',size(value),'data',value(:)');
end
function restore(c, original)
c.polymerizedRegions=original;
c.invalidate();
end
