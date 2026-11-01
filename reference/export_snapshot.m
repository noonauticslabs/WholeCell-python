function export_snapshot(sim, filename)
% Portable stored-state export. No dependent getters evaluated generically.
oldWarning=warning('off','MATLAB:structOnObject');restoreWarning=onCleanup(@() warning(oldWarning));
root=struct('schema_version',2,'source_commit','6cdee6b355aa0f5ff2953b1ab356eea049108e07','matlab_version',version,'objects',{{}},'errors',{{}});
objs={sim,sim.compartment,sim.gene};
ids={'simulation','constant:compartment','constant:gene'};
for i=1:numel(sim.states), objs{end+1}=sim.states{i};ids{end+1}=['state:' regexprep(sim.states{i}.wholeCellModelID,'^State_','')];end
for i=1:numel(sim.processes), objs{end+1}=sim.processes{i};ids{end+1}=['process:' regexprep(sim.processes{i}.wholeCellModelID,'^Process_','')];end
for i=1:numel(objs)
 obj=objs{i};mc=metaclass(obj);fields=struct();
 for j=1:numel(mc.PropertyList)
  prop=mc.PropertyList(j);
  if prop.Dependent || ~isequal(prop.GetAccess,'public'),continue;end
  try,fields.(prop.Name)=pack(obj.(prop.Name));catch e,root.errors{end+1}=struct('object',ids{i},'field',prop.Name,'error',e.message);end
 end
 root.objects{end+1}=struct('id',ids{i},'class',class(obj),'fields',fields);
end
root.complete=isempty(root.errors);
f=fopen(filename,'w');assert(f>=0);clean=onCleanup(@() fclose(f));fwrite(f,jsonencode(root),'char');
fprintf('SNAPSHOT_EXPORTED objects=%d errors=%d file=%s\n',numel(root.objects),numel(root.errors),filename);

function out=pack(v)
 if isa(v,'edu.stanford.covert.util.SparseMat')
  sv=struct(v);
  out=struct('kind','sparse','class',class(v),'subs',pack(sv.subs),'vals',pack(sv.vals),'shape',double(sv.siz));
  if isa(v,'edu.stanford.covert.util.CircularSparseMat'),out.circularDimensions=pack(sv.circularDims);end
 elseif isa(v,'edu.stanford.covert.cell.sim.constant.ChromosomeSequence')
  out=struct('kind','chromosome_sequence','positive_strand',reshape(v(:,1),1,[]));
 elseif isa(v,'edu.stanford.covert.util.RandStream')
  out=struct('kind','rng','type',v.type,'seed',pack(v.seed),'state',pack(v.state));
 elseif isa(v,'function_handle')
  out=struct('kind','matlab_function','expression',func2str(v));
 elseif isnumeric(v) || islogical(v)
  if issparse(v), [ii,jj,vv]=find(v);out=struct('kind','sparse','class','MATLAB.sparse','subs',pack([ii jj]),'vals',pack(vv),'shape',double(size(v)));return;end
  out=struct('kind','array','dtype',class(v),'shape',double(size(v)),'data',reshape(v,1,[]));
  if ~isreal(v),out.real=reshape(real(v),1,[]);out.imag=reshape(imag(v),1,[]);out=rmfield(out,'data');end
  if any(~isfinite(v(:))),out.nonfinite_indices=find(~isfinite(v(:)))';out.nonfinite_values=arrayfun(@(x) num2str(x),v(~isfinite(v)),'UniformOutput',false);out.data(~isfinite(out.data))=0;end
 elseif ischar(v)
  out=struct('kind','char','shape',double(size(v)),'data',reshape(v,1,[]));
 elseif iscell(v)
  data=cell(1,numel(v));for k=1:numel(v),data{k}=pack(v{k});end
  out=struct('kind','cell','shape',double(size(v)),'data',{data});
 elseif isstruct(v)
  data=cell(1,numel(v));names=fieldnames(v);
  for k=1:numel(v),d=struct();for z=1:numel(names),d.(names{z})=pack(v(k).(names{z}));end;data{k}=d;end
  out=struct('kind','struct','shape',double(size(v)),'data',{data});
 elseif isa(v,'edu.stanford.covert.cell.sim.SimulationStateSideEffect') || isa(v,'edu.stanford.covert.cell.sim.SimulationStateSideEffectItem')
  data=cell(1,numel(v));names=properties(v);for k=1:numel(v),d=struct();for z=1:numel(names),d.(names{z})=pack(v(k).(names{z}));end;data{k}=d;end
  out=struct('kind','value_objects','class',class(v),'shape',double(size(v)),'data',{data});
 elseif isobject(v)
  for k=1:numel(objs)
   if isa(v,'handle') && isscalar(v) && strcmp(class(v),class(objs{k})) && v==objs{k},out=struct('kind','reference','id',ids{k});return;end
  end
  error('Snapshot:unsupportedObject','No export encoding for class %s',class(v));
 else,error('Snapshot:unsupportedType','No export encoding for %s',class(v));end
end
end
