import bnlearn as bn
import os

# 1. create output directory
output_dir = 'data/raw/bnlearn'
os.makedirs(output_dir, exist_ok=True)
print(f"All files will be saved in '{output_dir}/'.\n")

# 2. list of bnlearn example datasets (hardcoded)
dataset_list = ['sprinkler', 'alarm', 'andes', 'asia', 'sachs', 'water']
print(f"Processing {len(dataset_list)} datasets.")
print(dataset_list)
print("-" * 30)

# 3. iterate datasets and save data and graph
for i, name in enumerate(dataset_list):
    print(f"[{i+1}/{len(dataset_list)}] Processing '{name}'...")
    
    try:
        # create per-dataset directory
        dataset_dir = os.path.join(output_dir, name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        # (1) load dataset
        df = bn.import_example(data=name)
        
        # (2) save raw data as CSV
        csv_path = os.path.join(dataset_dir, 'data.csv')
        df.to_csv(csv_path, index=False)
        
        # (3) learn graph structure from data (Hill-Climbing)
        # 'hc' (Hill-Climbing) is a common structure learning algorithm
        model = bn.structure_learning.fit(df, methodtype='hc')
        
        # (4) save learned model (graph + params) as pickle
        model_path = os.path.join(dataset_dir, 'model')
        bn.save(model, filepath=model_path, overwrite=True)
        
        # (5) save graph visualization as PNG
        # bn.plot_graphviz returns a graphviz object
        G = bn.plot_graphviz(model)
        graph_path = os.path.join(dataset_dir, 'graph')
        # save file via .render() (view=False suppresses window)
        G.render(filename=graph_path, format='png', view=False, cleanup=True)
        
        print(f"  -> '{name}' done: CSV, model, and PNG saved.")

    except Exception as e:
        # some datasets may require preprocessing or specific parameters
        print(f"  -> ERROR: failed processing '{name}': {e}")

print("-" * 30)
print(f"All done. Check the '{output_dir}' directory.")