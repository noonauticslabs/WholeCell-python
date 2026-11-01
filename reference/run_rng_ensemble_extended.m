function run_rng_ensemble_extended(checkpoint_file, output_file)
% Same biological t60 checkpoint, independently reseeded native MATLAB streams.
seeds = 101:120;
rows = zeros(numel(seeds), 8);
for i = 1:numel(seeds)
    loaded = load(checkpoint_file,'sim');
    sim = loaded.sim; clear loaded;
    assert(sim.state('Time').values == 60);
    sim.applyOptions('seed', seeds(i)); sim.seedRandStream();
    for j=1:60, sim.evolveState(); end
    [~, lengths] = find(sim.state('Chromosome').polymerizedRegions);
    rows(i,:) = [seeds(i), sim.state('Time').values, sum(sim.state('Mass').cellDry(:)), sim.state('MetabolicReaction').growth, sum(sim.state('Rna').counts(:)), sum(sim.state('ProteinMonomer').counts(:)), sum(sim.state('ProteinComplex').counts(:)), sum(lengths)];
    fprintf('ENSEMBLE_ROW %d\n',seeds(i));
    clear sim;
end
result = struct('description','Common MATLAB t60 biological state; reseeded streams; 60 second continuation', 'matlab_version',version,'columns',{{'seed','time_s','dry_mass_g','growth_per_s','rna_count','protein_monomer_count','protein_complex_count','polymerized_nucleotides'}},'rows',rows);
f=fopen(output_file,'w');fwrite(f,jsonencode(result),'char');fclose(f);
fprintf('ENSEMBLE_COMPLETE\n');
end
